import datetime
import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Optional, Literal

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import polars as pl
import requests
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import utils as ut

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Register validation for all numeric inputs
for input_id in ['squat-input', 'bench-input', 'deadlift-input', 'bodyweight-input']:
    ut.register_numeric_validation(app, input_id)

# Initialize global variables for optimization
current_context = {}

# Data paths
DATA_DIR = Path("data")
PARQUET_FILE = DATA_DIR / "openpowerlifting.parquet"
LAST_UPDATED_FILE = DATA_DIR / "last_updated.txt"
CSV_URL = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

def check_for_updates() -> bool:
    """Check if we need to update the dataset"""
    if not PARQUET_FILE.exists():
        return True
    if not LAST_UPDATED_FILE.exists():
        return True
    with LAST_UPDATED_FILE.open(mode='r') as f:
        last_updated_str = f.read().strip()
    last_updated = datetime.datetime.fromisoformat(last_updated_str)
    now = datetime.datetime.now()
    return (now - last_updated).days >= 1

def download_and_process_data() -> pl.LazyFrame:
    """Download and process OpenPowerlifting data using an explicit schema."""
    logger.info("Downloading latest OpenPowerlifting data...")
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to download data: {response.status_code}")
    
    # Extract the CSV file from the ZIP archive
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise Exception("No CSV file found in the ZIP archive")
        csv_file = csv_files[0]
        logger.info(f"Extracting {csv_file}...")
        z.extract(csv_file, DATA_DIR)
        csv_path = DATA_DIR / csv_file
        
        # Process the extracted data
        return process_powerlifting_data(csv_path)

def process_powerlifting_data(csv_path: Path) -> pl.LazyFrame:
    """Process powerlifting data from a CSV file."""
    # Define the explicit schema for the desired columns
    schema = {
        'Name': pl.Utf8,
        'Sex': pl.Utf8,  # Changed from Categorical to Utf8 to avoid Polars panic
        'Equipment': pl.Utf8,  # Changed from Categorical to Utf8
        'BodyweightKg': pl.Float32,
        'Age': pl.Float32,
        'Best3SquatKg': pl.Float32,
        'Best3BenchKg': pl.Float32,
        'Best3DeadliftKg': pl.Float32,
        'TotalKg': pl.Float32
    }
    
    logger.info("Processing data using the predefined schema...")
    base_lf = (
        pl.scan_csv(
            csv_path,
            schema_overrides=schema,
            infer_schema=False
        )
        .fill_null(strategy="zero")
        .filter(
            (pl.col('Best3SquatKg') > 0) |
            (pl.col('Best3BenchKg') > 0) |
            (pl.col('Best3DeadliftKg') > 0) |
            (pl.col('TotalKg') > 0)
        )
    )
    
    # Calculate weight class
    logger.info("Calculating weight classes...")
    base_lf = base_lf.with_columns([
        ut.calculate_ipf_weight_class(
            pl.col('BodyweightKg'),
            pl.col('Sex'),
            pl.col('Age')
        ).alias('WeightClassKg')
    ])
    
    # Process each lift type
    lift_types = [
        {
            'name': 'Squat',
            'value_col': 'Best3SquatKg',
            'filter': pl.col('Best3SquatKg') > 0
        },
        {
            'name': 'Bench',
            'value_col': 'Best3BenchKg',
            'filter': pl.col('Best3BenchKg') > 0
        },
        {
            'name': 'Deadlift',
            'value_col': 'Best3DeadliftKg',
            'filter': pl.col('Best3DeadliftKg') > 0
        },
        {
            'name': 'Total',
            'value_col': 'TotalKg',
            'filter': pl.col('TotalKg') > 0
        }
    ]
    
    # Process each lift type and store results
    lift_frames = {}
    
    for lift in lift_types:
        lift_name = lift['name']
        value_col = lift['value_col']
        filter_expr = lift['filter']
        
        # Create a lift-specific lazy frame
        cols_to_select = ['Name']
        if lift_name == 'Squat':  # Only squat needs Sex for joins
            cols_to_select.append('Sex')
            
        cols_to_select.extend([
            value_col,
            'Equipment',
            'BodyweightKg',
            'WeightClassKg'
        ])
        
        lift_lf = (
            base_lf
            .filter(filter_expr)
            .select(cols_to_select)
        )
        
        # Window function to find max values
        window = pl.col('Name')
        max_col_alias = f"Max{lift_name}"
        
        lift_lf = (
            lift_lf
            .with_columns([
                pl.max(value_col).over(window).alias(max_col_alias)
            ])
            .filter(pl.col(value_col) == pl.col(max_col_alias))
        )
        
        # Select and rename columns
        select_cols = ['Name']
        if lift_name == 'Squat':
            select_cols.append('Sex')
            
        select_cols.extend([
            pl.col(value_col),
            pl.col('Equipment').alias(f"{lift_name}Equipment"),
            pl.col('BodyweightKg').alias(f"{lift_name}BodyweightKg"),
            pl.col('WeightClassKg').alias(f"{lift_name}WeightClassKg")
        ])
        
        lift_lf = lift_lf.select(select_cols)
        
        # Group by Name (and Sex for squat)
        group_cols = ['Name']
        if lift_name == 'Squat':
            group_cols.append('Sex')
            
        agg_cols = [
            pl.max(value_col).alias(value_col),
            pl.first(f"{lift_name}Equipment").alias(f"{lift_name}Equipment"),
            pl.first(f"{lift_name}BodyweightKg").alias(f"{lift_name}BodyweightKg"),
            pl.first(f"{lift_name}WeightClassKg").alias(f"{lift_name}WeightClassKg")
        ]
        
        lift_lf = lift_lf.group_by(group_cols).agg(agg_cols)
        lift_frames[lift_name.lower()] = lift_lf
    
    # Join all the LazyFrames
    # Start with squat as the base
    result_lf = lift_frames['squat']
    
    # Join with bench, deadlift, and total
    for lift_name in ['bench', 'deadlift', 'total']:
        result_lf = result_lf.join(
            lift_frames[lift_name], 
            on='Name', 
            how='full', 
            suffix=f'_{lift_name}'
        )
    
    # Fill nulls for numeric columns
    result_lf = result_lf.with_columns([
        pl.when(pl.col('Best3SquatKg').is_null()).then(0).otherwise(pl.col('Best3SquatKg')).alias('Best3SquatKg'),
        pl.when(pl.col('Best3BenchKg').is_null()).then(0).otherwise(pl.col('Best3BenchKg')).alias('Best3BenchKg'),
        pl.when(pl.col('Best3DeadliftKg').is_null()).then(0).otherwise(pl.col('Best3DeadliftKg')).alias('Best3DeadliftKg'),
        pl.when(pl.col('TotalKg').is_null()).then(0).otherwise(pl.col('TotalKg')).alias('TotalKg')
    ])
    
    # Calculate Wilks scores
    result_lf = ut.calculate_wilks_scores(result_lf)
    
    # Log column availability for debugging
    logger.info("Columns available after processing (before writing to parquet):")
    try:
        logger.info(f"Columns: {result_lf.collect_schema().names()}")
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")
    
    # Instead of using sink_parquet directly, we'll collect and then write
    try:
        logger.info(f"Collecting and saving processed data to {PARQUET_FILE}...")
        collected_df = result_lf.collect()
        collected_df.write_parquet(PARQUET_FILE, compression="zstd", compression_level=9)
    except Exception as e:
        logger.error(f"Error saving parquet file: {e}")
        # Create a fallback version of the file using a simpler approach
        try:
            logger.info("Trying alternative approach to save data...")
            # Convert categorical columns to strings
            for col in collected_df.columns:
                if collected_df[col].dtype == pl.Categorical:
                    collected_df = collected_df.with_columns([
                        pl.col(col).cast(pl.Utf8).alias(col)
                    ])
            collected_df.write_parquet(PARQUET_FILE, compression="zstd", compression_level=9)
            logger.info("Successfully saved data with alternative approach")
        except Exception as e2:
            logger.error(f"Alternative save method also failed: {e2}")
            # Last resort - try CSV format instead
            logger.info("Attempting to save as CSV instead...")
            csv_path = DATA_DIR / "openpowerlifting_backup.csv"
            collected_df.write_csv(csv_path)
            logger.info(f"Saved data as CSV to {csv_path}")
    
    with LAST_UPDATED_FILE.open(mode='w') as f:
        f.write(datetime.datetime.now().isoformat())
    
    logger.info("Data update complete!")
    return result_lf

def load_data() -> pl.LazyFrame:
    """Load the data, downloading it first if necessary"""
    if check_for_updates():
        return download_and_process_data()
    else:
        logger.info("Loading cached data...")
        try:
            return pl.scan_parquet(PARQUET_FILE)
        except Exception as e:
            logger.error(f"Error loading parquet file: {e}")
            
            # Check if we have a CSV backup
            csv_backup = DATA_DIR / "openpowerlifting_backup.csv"
            if csv_backup.exists():
                logger.info(f"Trying to load from CSV backup: {csv_backup}")
                try:
                    return pl.scan_csv(csv_backup)
                except Exception as csv_e:
                    logger.error(f"Error loading CSV backup: {csv_e}")
            
            # If we get here, we need to re-download and process the data
            logger.info("Re-downloading and processing data...")
            return download_and_process_data()

@ut.advanced_cached_figure
def create_histogram(df: pl.DataFrame, equipment: list[str], lift: str, user_value: Optional[float]=None, use_wilks: bool=False, sample_size: int=10000) -> go.Figure:
    """Create a histogram for the specified lift with improved performance."""
    if df.height == 0:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    try:
        column, wilks_column, _ = ut.get_lift_columns(lift)
    except ValueError as e:
        logger.error(f"Unknown lift type: {e}")
        return go.Figure()

    # Check if the Wilks column exists when requested
    if use_wilks and wilks_column not in df.columns:
        logger.warning(f"{wilks_column} not found in dataframe. Falling back to regular values.")
        use_wilks = False
    
    # Fast filtering with expression
    plot_df = ut.get_specific_filter(df, column, lift, equipment)
    
    if plot_df.height == 0:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No positive {lift} values found for the selected filters",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Check how many unique sexes we have
    unique_sexes: list[str] = plot_df['Sex'].unique().to_list()
    has_multiple_sexes = len(unique_sexes) > 1
    
    # More efficient sampling approach
    plot_df = ut.sample_data(plot_df, lift, sample_size, create_histogram)

    # Create the figure
    fig = go.Figure()
    plot_column = wilks_column if use_wilks else column
    
    # Optimize for the case of multiple sexes
    if has_multiple_sexes:
        # Pre-compute bin settings for consistent bins across traces
        all_values = plot_df[plot_column]
        if len(all_values) > 0:
            value_min = all_values.min()
            value_max = all_values.max()
            # Ensure sensible range even with outliers
            value_range = max(value_max - value_min, 1)  # Avoid division by zero
            # Create bins with padding
            bin_size = value_range / 50
            bin_min = max(0, value_min - bin_size)
            bin_max = value_max + bin_size
            bins = np.linspace(bin_min, bin_max, 51)
        else:
            bins = 50  # Fallback
            
        # Create separate traces for each sex
        for sex in unique_sexes:
            sex_data = plot_df.filter(pl.col('Sex') == sex)
            if sex_data.height > 0:
                values = sex_data[plot_column].to_numpy()
                
                # Choose color based on sex
                color = 'blue' if sex == 'M' else 'pink'
                name = 'Male' if sex == 'M' else 'Female'
                
                fig.add_trace(go.Histogram(
                    x=values,
                    xbins=dict(
                        start=bins[0],
                        end=bins[-1],
                        size=(bins[-1] - bins[0]) / 50
                    ),
                    marker_color=color,
                    opacity=0.6,
                    name=name
                ))
        
        fig.update_layout(barmode='overlay')
    else:
        # Single sex - simpler approach
        values = plot_df[plot_column].to_numpy()
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=50,
            marker_color='#3182bd',
            opacity=0.75
        ))
    
    # Add user value line if provided
    if user_value is not None and user_value > 0:
        fig.add_vline(
            x=user_value, 
            line_width=2, 
            line_color="darkred",
            annotation_text="Your lift",
            annotation_position="top right"
        )
        
        # Calculate percentile if possible
        if len(values) > 0:
            percentile = np.sum(values <= user_value) / len(values) * 100
            fig.add_annotation(
                x=user_value,
                y=0,
                yshift=10,
                text=f"Your lift: {percentile:.1f}th percentile",
                showarrow=False,
                font=dict(color="darkred")
            )
    
    # Customize layout
    title = f"{lift} Wilks Histogram" if use_wilks else f"{lift} Histogram"
    x_title = f"{lift} Wilks Score" if use_wilks else f"{lift} (kg)"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Frequency",
        plot_bgcolor='white',
        bargap=0.1,
        showlegend=has_multiple_sexes,
        # Improved layout for better readability
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

@ut.advanced_cached_figure
def create_scatter_plot(df: pl.DataFrame,equipment: list[str],  lift: str, use_wilks: bool=False, 
                       user_bodyweight: Optional[float]=None, user_lift: Optional[float]=None, 
                       sample_size: int=10000) -> go.Figure:
    """Create a scatter plot of the lift vs. bodyweight with improved performance."""
    if df.height == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    try:
        column, wilks_column, bodyweight_column = ut.get_lift_columns(lift, bdy=True)
    except ValueError as e:
        logger.error(f"Unknown lift type: {e}")
        return go.Figure()
    
    # Check if the Wilks column exists when requested
    if use_wilks and wilks_column not in df.columns:
        logger.warning(f"{wilks_column} not found in dataframe. Falling back to regular values.")
        use_wilks = False
    
    # Filter for rows with positive lift and bodyweight values
    plot_df = ut.get_specific_filter(df, column, lift, equipment, bodyweight_column)
    
    if plot_df.height == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No positive {lift} values found for the selected filters",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Efficient sampling strategy
    plot_df = ut.sample_data(plot_df, lift, sample_size, create_scatter_plot)

    # Determine which y-column to use
    y_column = wilks_column if use_wilks else column
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Optimize for common case of two sexes (M/F)
    for sex, color, name in [('M', 'blue', 'Male Lifters'), ('F', 'pink', 'Female Lifters')]:
        sex_data = plot_df.filter(pl.col('Sex') == sex)
        if sex_data.height > 0:
            # Extract data all at once for better performance
            x_values = sex_data[bodyweight_column].to_numpy()
            y_values = sex_data[y_column].to_numpy()
            
            # Create more efficient scatter plot
            # Use smaller marker size and increased transparency for better visualization
            fig.add_trace(go.Scattergl(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=4,  # Smaller markers for better density visualization
                    color=color,
                    opacity=0.5,  # More transparency for overlapping points
                    line=dict(width=0)  # Remove marker borders for performance
                ),
                name=name
            ))
    
    # Handle other sexes if present (rare but possible)
    other_sexes = [s for s in plot_df['Sex'].unique().to_list() if s not in ['M', 'F']]
    for sex in other_sexes:
        sex_data = plot_df.filter(pl.col('Sex') == sex)
        if sex_data.height > 0:
            fig.add_trace(go.Scattergl(
                x=sex_data[bodyweight_column].to_numpy(),
                y=sex_data[y_column].to_numpy(),
                mode='markers',
                marker=dict(size=4, opacity=0.5),
                name=f'{sex} Lifters'
            ))
    
    # Add user's data point if provided
    if user_bodyweight is not None and user_lift is not None and user_bodyweight > 0 and user_lift > 0:
        fig.add_trace(go.Scattergl(
            x=[user_bodyweight],
            y=[user_lift],
            mode='markers',
            marker=dict(
                size=12, 
                color='red', 
                symbol='star-triangle-up',
                line=dict(width=1, color='black')
            ),
            name='Your Lift'
        ))
        
        # Calculate and display percentile if possible
        if plot_df.height > 0:
            # Get values close to the user's bodyweight
            weight_range = 5  # kg
            similar_weight_lifters = plot_df.filter(
                (pl.col(bodyweight_column) >= user_bodyweight - weight_range) &
                (pl.col(bodyweight_column) <= user_bodyweight + weight_range)
            )
            
            if similar_weight_lifters.height > 0:
                y_values = similar_weight_lifters[y_column].to_numpy()
                percentile = np.sum(y_values <= user_lift) / len(y_values) * 100
                
                fig.add_annotation(
                    x=user_bodyweight,
                    y=user_lift,
                    text=f"{percentile:.1f}th percentile at your weight",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=40,
                    ay=-40
                )
    
    # Add trendlines
    if plot_df.height >= 10:
        for sex, color, name in [('M', 'rgba(0,0,255,0.7)', 'Male Trend'), ('F', 'rgba(255,105,180,0.7)', 'Female Trend')]:
            sex_data = plot_df.filter(pl.col('Sex') == sex)
            if sex_data.height >= 10:  # Only add trendline if enough data points
                x = sex_data[bodyweight_column].to_numpy()
                y = sex_data[y_column].to_numpy()
                
                # Simple linear fit
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_range = np.linspace(min(x), max(x), 100)
                    y_pred = slope * x_range + intercept
                    
                    fig.add_trace(go.Scattergl(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        line=dict(color=color, width=2, dash='dash'),
                        name=name,
                        showlegend=True
                    ))
                except Exception:
                    # Skip trendline if fit fails
                    pass
    
    # Customize layout
    y_title = f"{lift} Wilks Score" if use_wilks else f"{lift} (kg)"
    title = f"{lift} Wilks Score vs. Bodyweight" if use_wilks else f"{lift} vs. Bodyweight"
    
    fig.update_layout(
        title=title,
        xaxis_title="Bodyweight (kg)",
        yaxis_title=y_title,
        plot_bgcolor='white',
        hovermode='closest',
        # Improved layout for better readability
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="right", 
            x=0.99,
            bordercolor="LightGrey",
            borderwidth=1
        )
    )
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    return fig


# ---------------- Callback Definitions ----------------
@app.callback(
    Output('weight-class-dropdown', 'options'),
    [Input('sex-filter', 'value')]
)
def update_weight_class_options(sex: Literal['M', 'F', 'All']):
    return ut.get_weight_class_options(sex)

@app.callback(
    [Output('squat-histogram', 'figure'),
     Output('squat-wilks-histogram', 'figure'),
     Output('squat-scatter', 'figure'),
     Output('squat-wilks-scatter', 'figure'),
     Output('bench-histogram', 'figure'),
     Output('bench-wilks-histogram', 'figure'),
     Output('bench-scatter', 'figure'),
     Output('bench-wilks-scatter', 'figure'),
     Output('deadlift-histogram', 'figure'),
     Output('deadlift-wilks-histogram', 'figure'),
     Output('deadlift-scatter', 'figure'),
     Output('deadlift-wilks-scatter', 'figure'),
     Output('total-histogram', 'figure'),
     Output('total-wilks-histogram', 'figure'),
     Output('total-scatter', 'figure'),
     Output('total-wilks-scatter', 'figure'),
     Output('last-updated-text', 'children')],
    [Input('update-button', 'n_clicks'),
     Input('squat-input', 'value'),
     Input('bench-input', 'value'),
     Input('deadlift-input', 'value'),
     Input('bodyweight-input', 'value'),
     Input('sex-filter', 'value'),
     Input('equipment-filter', 'value'),
     Input('weight-class-dropdown', 'value'),
     Input('units', 'value')]
)
def update_all_figures(n_clicks: Optional[int], squat: Optional[float], bench: Optional[float], deadlift: Optional[float], bodyweight: Optional[float], sex: str, equipment: list[str], weight_class: str, units: str):
    """Optimized unified callback that handles all input changes."""
    # Get context to determine which input triggered the callback
    # Initialize filtered_df_cache if not already in globals
    if 'filtered_df_cache' not in globals():
        globals()['filtered_df_cache'] = {
            'key': None,
            'df': None,
            'row_count': 0
        }

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Skip initial call if no trigger
    if triggered_id == '' and n_clicks is None:
        raise PreventUpdate
    
    # Metric/Imperial conversion if needed
    conversion_factor = 2.20462  # kg to lbs
    converted_squat, converted_bench, converted_deadlift, converted_bodyweight = squat, bench, deadlift, bodyweight
    
    if units == 'imperial':
        # Convert imperial inputs to metric for processing
        if squat is not None:
            converted_squat = squat / conversion_factor
        if bench is not None:
            converted_bench = bench / conversion_factor
        if deadlift is not None:
            converted_deadlift = deadlift / conversion_factor
        if bodyweight is not None:
            converted_bodyweight = bodyweight / conversion_factor
    
    # Generate a cache key based on filter parameters
    filter_cache_key = f"{sex}_{','.join(sorted(equipment) if equipment else [])}_{weight_class}_{units}"
    
    # Determine if we need a full data reload or just figure updates
    need_full_update = (
        triggered_id == 'update-button' or
        triggered_id == 'sex-filter' or
        triggered_id == 'equipment-filter' or
        triggered_id == 'weight-class-dropdown' or
        triggered_id == 'units' or
        globals()['filtered_df_cache']['key'] != filter_cache_key or
        globals()['filtered_df_cache']['df'] is None
    )
    
    # For full updates, load and filter the data
    if need_full_update:
        logger.info(f"Full data update triggered by {triggered_id}")
        
        # Load data if needed
        if 'lf' not in globals() or globals()['lf'] is None:
            try:
                logger.info("Loading data...")
                start_time = time.time()
                globals()['lf'] = load_data()
                logger.info(f"Data loading completed in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                # Return empty figures with error message
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error loading data: {e}", showarrow=False, font=dict(size=14, color="red"))
                return [empty_fig] * 16 + [f"Error loading data: {e}"]
        
        # Filter the data
        try:
            start_time = time.time()
            filtered_lf = ut.filter_data(globals()['lf'], sex, equipment, weight_class)
            logger.info(f"Data filtering completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            # Return empty figures with error message
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=f"Error filtering data: {e}", showarrow=False, font=dict(size=14, color="red"))
            return [empty_fig] * 16 + [f"Error filtering data: {e}"]
        
        # Handle unit conversion if needed
        if units == 'imperial':
            conversion_cols = [
                (pl.col('Best3SquatKg') * conversion_factor).alias('Best3SquatKg'),
                (pl.col('Best3BenchKg') * conversion_factor).alias('Best3BenchKg'),
                (pl.col('Best3DeadliftKg') * conversion_factor).alias('Best3DeadliftKg'),
                (pl.col('TotalKg') * conversion_factor).alias('TotalKg'),
                (pl.col('SquatBodyweightKg') * conversion_factor).alias('SquatBodyweightKg'),
                (pl.col('BenchBodyweightKg') * conversion_factor).alias('BenchBodyweightKg'),
                (pl.col('DeadliftBodyweightKg') * conversion_factor).alias('DeadliftBodyweightKg'),
                (pl.col('TotalBodyweightKg') * conversion_factor).alias('TotalBodyweightKg')
            ]
            filtered_lf = filtered_lf.with_columns(conversion_cols)
        
        # Execute the query and collect once after all filters
        try:
            start_time = time.time()
            filtered_df = filtered_lf.collect()
            logger.info(f"Collected filtered data in {time.time() - start_time:.2f}s - {filtered_df.height} rows")
        except Exception as e:
            logger.error(f"Error collecting filtered data: {e}")
            # Better error message with schema information for debugging
            try:
                schema = globals()['lf'].collect_schema()
                schema_info = ", ".join(schema.names())
                err_msg = f"Error collecting filtered data: {e}. Schema: {schema_info[:100]}..."
            except Exception as e:
                err_msg = f"Error collecting filtered data: {e}"
                
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=err_msg, showarrow=False, font=dict(size=14, color="red"))
            return [empty_fig] * 16 + [err_msg]
        
        # Update the filtered data cache
        globals()['filtered_df_cache'] = {
            'key': filter_cache_key,
            'df': filtered_df,
            'row_count': filtered_df.height
        }
    else:
        # Use the cached filtered data
        logger.info(f"Using cached data for {triggered_id} - {globals()['filtered_df_cache']['row_count']} rows")
        filtered_df = globals()['filtered_df_cache']['df']
    
    # Calculate user total if all lifts are provided
    user_total = None
    if all(x is not None for x in [converted_squat, converted_bench, converted_deadlift]):
        user_total = converted_squat + converted_bench + converted_deadlift
    
    # Calculate user's weight class if bodyweight is provided
    user_weight_class = None
    if converted_bodyweight is not None and sex is not None and sex != 'All':
        user_weight_class = ut.calculate_user_weight_class(converted_bodyweight, sex)
    
    # Determine which figures need to be updated based on the trigger
    # This optimized approach only updates the necessary figures
    figure_updates = {
        'Squat': {
            'histogram': triggered_id in ('update-button', 'squat-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_histogram': triggered_id in ('update-button', 'squat-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units', 'bodyweight-input') or need_full_update,
            'scatter': triggered_id in ('update-button', 'squat-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_scatter': triggered_id in ('update-button', 'squat-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
        },
        'Bench': {
            'histogram': triggered_id in ('update-button', 'bench-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_histogram': triggered_id in ('update-button', 'bench-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units', 'bodyweight-input') or need_full_update,
            'scatter': triggered_id in ('update-button', 'bench-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_scatter': triggered_id in ('update-button', 'bench-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
        },
        'Deadlift': {
            'histogram': triggered_id in ('update-button', 'deadlift-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_histogram': triggered_id in ('update-button', 'deadlift-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units', 'bodyweight-input') or need_full_update,
            'scatter': triggered_id in ('update-button', 'deadlift-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_scatter': triggered_id in ('update-button', 'deadlift-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
        },
        'Total': {
            'histogram': triggered_id in ('update-button', 'squat-input', 'bench-input', 'deadlift-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_histogram': triggered_id in ('update-button', 'squat-input', 'bench-input', 'deadlift-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units', 'bodyweight-input') or need_full_update,
            'scatter': triggered_id in ('update-button', 'squat-input', 'bench-input', 'deadlift-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
            'wilks_scatter': triggered_id in ('update-button', 'squat-input', 'bench-input', 'deadlift-input', 'bodyweight-input', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units') or need_full_update,
        }
    }
    
    # Map lift types to their values
    lift_values = {
        'Squat': converted_squat,
        'Bench': converted_bench,
        'Deadlift': converted_deadlift,
        'Total': user_total
    }
    
    # Initialize results with no_update placeholders
    results = [dash.no_update] * 16
    
    # Count how many figures we'll actually generate
    figures_to_generate = sum(
        1 for lift in figure_updates 
        for fig_type in figure_updates[lift] 
        if figure_updates[lift][fig_type]
    )
    
    if figures_to_generate > 0:
        logger.info(f"Generating {figures_to_generate} figures")
    
    # Figure indexing map to match the expected output order
    figure_indices = {
        'Squat': {
            'histogram': 0,
            'wilks_histogram': 1,
            'scatter': 2,
            'wilks_scatter': 3
        },
        'Bench': {
            'histogram': 4,
            'wilks_histogram': 5,
            'scatter': 6,
            'wilks_scatter': 7
        },
        'Deadlift': {
            'histogram': 8,
            'wilks_histogram': 9,
            'scatter': 10,
            'wilks_scatter': 11
        },
        'Total': {
            'histogram': 12,
            'wilks_histogram': 13,
            'scatter': 14,
            'wilks_scatter': 15
        }
    }
    
    # Generate only the figures that need updating
    for lift_type in ['Squat', 'Bench', 'Deadlift', 'Total']:
        user_value = lift_values[lift_type]
        
        # For calculating user's Wilks scores
        if sex and sex != 'All' and converted_bodyweight is not None and user_value is not None:
            user_wilks = ut.get_wilks_value(user_value, converted_bodyweight, sex)
        else:
            user_wilks = None
        
        # Generate regular histogram if needed
        if figure_updates[lift_type]['histogram']:
            idx = figure_indices[lift_type]['histogram']
            try:
                start_time = time.time()
                results[idx] = create_histogram(filtered_df, equipment, lift_type, user_value)
                logger.info(f"Generated {lift_type} histogram in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating {lift_type} histogram: {e}")
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error generating {lift_type} histogram: {e}", showarrow=False, font=dict(size=14, color="red"))
                results[idx] = empty_fig
        
        # Generate Wilks histogram if needed
        if figure_updates[lift_type]['wilks_histogram'] and sex != 'All':
            idx = figure_indices[lift_type]['wilks_histogram']
            try:
                start_time = time.time()
                results[idx] = create_histogram(filtered_df, equipment, lift_type, user_wilks, use_wilks=True)
                logger.info(f"Generated {lift_type} Wilks histogram in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating {lift_type} Wilks histogram: {e}")
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error generating {lift_type} Wilks histogram: {e}", showarrow=False, font=dict(size=14, color="red"))
                results[idx] = empty_fig
        
        # Generate scatter plot if needed
        if figure_updates[lift_type]['scatter']:
            idx = figure_indices[lift_type]['scatter']
            try:
                start_time = time.time()
                results[idx] = create_scatter_plot(
                    filtered_df, equipment, lift_type, 
                    user_bodyweight=converted_bodyweight,
                    user_lift=user_value
                )
                logger.info(f"Generated {lift_type} scatter plot in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating {lift_type} scatter plot: {e}")
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error generating {lift_type} scatter plot: {e}", showarrow=False, font=dict(size=14, color="red"))
                results[idx] = empty_fig
        
        # Generate Wilks scatter plot if needed
        if figure_updates[lift_type]['wilks_scatter'] and sex != 'All':
            idx = figure_indices[lift_type]['wilks_scatter']
            try:
                start_time = time.time()
                results[idx] = create_scatter_plot(
                    filtered_df, equipment, lift_type, 
                    use_wilks=True,
                    user_bodyweight=converted_bodyweight,
                    user_lift=user_wilks
                )
                logger.info(f"Generated {lift_type} Wilks scatter plot in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating {lift_type} Wilks scatter plot: {e}")
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Error generating {lift_type} Wilks scatter plot: {e}", showarrow=False, font=dict(size=14, color="red"))
                results[idx] = empty_fig
    
    # Update last updated text if needed
    if need_full_update:
        last_updated = "Unknown"
        if LAST_UPDATED_FILE.exists():
            with LAST_UPDATED_FILE.open(mode='r') as f:
                last_updated_str = f.read().strip()
                try:
                    last_updated = datetime.datetime.fromisoformat(last_updated_str).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
        
        sex_info = f" ({sex})" if sex != 'All' else ""
        equipment_info = f", Equipment: {', '.join(equipment)}" if equipment and len(equipment) > 0 else ""
        w_class = user_weight_class or weight_class
        weight_class_info = f", Weight Class: {w_class}" if w_class else ""
        
        row_count = globals()['filtered_df_cache']['row_count']
        unit_text = "lbs" if units == 'imperial' else "kg"
        last_updated_text = f"Data last updated: {last_updated} | Showing {row_count:,} lifters{sex_info}{equipment_info}{weight_class_info} | Units: {unit_text}"
    else:
        # No need to update text if not doing a full update
        last_updated_text = dash.no_update
    
    # Add the last updated text to the results
    results.append(last_updated_text)
    
    return results

@app.callback(
    Output('share-twitter', 'href'),
    [Input('update-button', 'n_clicks')],
    [State('squat-input', 'value'),
     State('bench-input', 'value'),
     State('deadlift-input', 'value'),
     State('bodyweight-input', 'value')]
)
def update_twitter_link(n_clicks: Optional[int], squat: Optional[float], bench: Optional[float], deadlift: Optional[float], bodyweight: Optional[float]) -> str:
    if None in [squat, bench, deadlift, bodyweight]:
        return "#"
    total = squat + bench + deadlift
    share_text = f"My powerlifting stats: Squat {squat}kg, Bench {bench}kg, Deadlift {deadlift}kg, Total {total}kg at {bodyweight}kg bodyweight. Check where you stand at our app!"
    return f"https://twitter.com/intent/tweet?text={share_text}"

@app.callback(
    Output('share-facebook', 'href'),
    [Input('update-button', 'n_clicks')]
)
def update_facebook_link(n_clicks: Optional[int]) -> str:
    return "https://www.facebook.com/sharer/sharer.php?u=https://powerlifting-visualizer.example.com"

@app.callback(
    Output('share-instagram', 'href'),
    [Input('update-button', 'n_clicks')]
)
def update_instagram_link(n_clicks: Optional[int]) -> str:
    return "#"

# ---------------- App Layout ----------------

app.layout = html.Div([
    html.H1("Powerlifting Data Visualization", className="mt-4 mb-4 text-center"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Filter Options", className="mb-3"),
                html.Label("Sex:"),
                dbc.RadioItems(
                    id='sex-filter',
                    options=[
                        {'label': 'Male', 'value': 'M'},
                        {'label': 'Female', 'value': 'F'},
                        {'label': 'All', 'value': 'All'}
                    ],
                    value='M',
                    className="mb-3",
                    inline=True
                ),
                html.Label("Units:"),
                dbc.RadioItems(
                    id='units',
                    options=[
                        {'label': 'Metric (kg)', 'value': 'metric'},
                        {'label': 'Imperial (lbs)', 'value': 'imperial'}
                    ],
                    value='metric',
                    className="mb-3",
                    inline=True
                ),
                html.Label("Weight Class:"),
                dcc.Dropdown(
                    id='weight-class-dropdown',
                    options=ut.get_weight_class_options('M'),  # Default to male options
                    value='all',
                    clearable=False,
                    className="mb-3"
                ),
                html.Label("Equipment:"),
                dbc.Checklist(
                    id='equipment-filter',
                    options=[
                        {'label': 'Raw', 'value': 'Raw'},
                        {'label': 'Wraps', 'value': 'Wraps'},
                        {'label': 'Single-ply', 'value': 'Single-ply'},
                        {'label': 'Multi-ply', 'value': 'Multi-ply'},
                        {'label': 'Unlimited', 'value': 'Unlimited'},
                        {'label': 'Straps', 'value': 'Straps'}
                    ],
                    value=['Raw'],
                    className="mb-3"
                ),
                html.Label("Your Lifts (kg):"),
                html.Div([
                    dbc.InputGroup([
                        dbc.InputGroupText("Squat"),
                        dbc.Input(id='squat-input', type='number', min=0, step=2.5, placeholder='kg', inputMode='decimal', pattern="[0-9]*[.]?[0-9]*")
                    ], className="mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Bench"),
                        dbc.Input(id='bench-input', type='number', min=0, step=2.5, placeholder='kg', inputMode='decimal', pattern="[0-9]*[.]?[0-9]*")
                    ], className="mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Deadlift"),
                        dbc.Input(id='deadlift-input', type='number', min=0, step=2.5, placeholder='kg', inputMode='decimal', pattern="[0-9]*[.]?[0-9]*")
                    ], className="mb-2"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Bodyweight"),
                        dbc.Input(id='bodyweight-input', type='number', min=0, step=0.1, placeholder='kg', inputMode='decimal', pattern="[0-9]*[.]?[0-9]*")
                    ], className="mb-3")
                ]),
                dbc.Button(
                    "Update Visualizations", 
                    id="update-button", 
                    color="primary", 
                    className="w-100 mb-3"
                ),
                html.Div([
                    html.P(id="last-updated-text", className="text-muted small")
                ])
            ], className="p-3 border rounded")
        ], width=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Squat", children=[
                    dbc.Row([dbc.Col([dcc.Graph(id='squat-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='squat-wilks-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='squat-scatter')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='squat-wilks-scatter')], width=12)])
                ]),
                dbc.Tab(label="Bench Press", children=[
                    dbc.Row([dbc.Col([dcc.Graph(id='bench-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='bench-wilks-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='bench-scatter')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='bench-wilks-scatter')], width=12)])
                ]),
                dbc.Tab(label="Deadlift", children=[
                    dbc.Row([dbc.Col([dcc.Graph(id='deadlift-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='deadlift-wilks-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='deadlift-scatter')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='deadlift-wilks-scatter')], width=12)])
                ]),
                dbc.Tab(label="Total", children=[
                    dbc.Row([dbc.Col([dcc.Graph(id='total-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='total-wilks-histogram')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='total-scatter')], width=12)]),
                    dbc.Row([dbc.Col([dcc.Graph(id='total-wilks-scatter')], width=12)])
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Button("Share to Twitter", id="share-twitter", color="info", className="mr-2"),
                        dbc.Button("Share to Facebook", id="share-facebook", color="primary", className="mr-2"),
                        dbc.Button("Share to Instagram", id="share-instagram", color="danger")
                    ], className="mt-3 d-flex justify-content-center gap-2")
                ], width=12)
            ])
        ], width=9)
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
