import datetime
import logging
import time
from typing import List, Literal, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from models import load_data, get_weight_class_options, get_wilks_value, calculate_user_weight_class
from services import config
from utils import empty_figure, register_numeric_validation, create_histogram, create_scatter_plot, filter_data, \
    get_metrics
from views import create_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get configuration
APP_NAME = config.get("app", "name") or "Powerlifting Data Visualization"
DEBUG_MODE = config.get("app", "debug") or False
CONVERSION_FACTOR = config.get("display", "weight_conversion_factor") or 2.20462  # kg to lbs

# Set up the app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = APP_NAME
server = app.server

# Register validation for all numeric inputs
for input_id in ['squat-input', 'bench-input', 'deadlift-input', 'bodyweight-input']:
    register_numeric_validation(app, input_id)

# Initialize the app layout
app.layout = create_layout()

# Initialize global variables for optimization
filtered_df_cache = {
    'key': None,
    'df': None,
    'row_count': 0
}

# ---------------- Callback Definitions ----------------

@app.callback(
    Output('weight-class-dropdown', 'options'),
    [Input('sex-filter', 'value')]
)
def update_weight_class_options(sex: Literal['M', 'F', 'All']):
    """Update weight class dropdown options based on selected sex."""
    return get_weight_class_options(sex)

@app.callback(
    Output('info-modal', 'is_open'),
    [Input('open-info-button', 'n_clicks'), Input('close-info-modal', 'n_clicks')],
    [State('info-modal', 'is_open')],
)
def toggle_info_modal(open_clicks: Optional[int], close_clicks: Optional[int], is_open: bool):
    """Toggle the information modal."""
    if open_clicks or close_clicks:
        return not is_open
    return is_open

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
def update_all_figures(n_clicks: Optional[int], squat: Optional[float], bench: Optional[float], 
                      deadlift: Optional[float], bodyweight: Optional[float], sex: str, 
                      equipment: List[str], weight_class: str, units: str):
    """Optimized unified callback that handles all input changes."""
    # Get context to determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Skip initial call if no trigger
    if triggered_id == '' and n_clicks is None:
        raise PreventUpdate
    
    # Metric/Imperial conversion if needed
    conversion_factor = CONVERSION_FACTOR
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
    
    #####################
    logger.info(f"Callback triggered by: {triggered_id}")
    logger.info(f"Current inputs: sex={sex}, bodyweight={bodyweight}, equipment={equipment}")
    logger.info(f"Context: {dash.callback_context.triggered}")
    #####################
    
    # Generate a cache key based on filter parameters
    filter_cache_key = f"{sex}_{','.join(sorted(equipment) if equipment else [])}_{weight_class}_{units}_{converted_bodyweight}"
    
    # Determine if we need a full data reload or just figure updates
    need_full_update = (
        triggered_id == 'update-button' or
        triggered_id == 'sex-filter' or
        triggered_id == 'equipment-filter' or
        triggered_id == 'weight-class-dropdown' or
        triggered_id == 'units' or
        triggered_id == 'bodyweight-input' or
        filtered_df_cache['key'] != filter_cache_key or
        filtered_df_cache['df'] is None
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
            filtered_lf = filter_data(globals()['lf'], sex, equipment, weight_class)
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

            # Ensure Wilks scores are calculated in the filtered data
            if filtered_df is not None and filtered_df.height > 0:
                # Check if Wilks columns exist
                wilks_columns = [col for col in filtered_df.columns if "Wilks" in col]
                if not wilks_columns:
                    logger.warning("No Wilks columns found, attempting to calculate them...")
                    try:
                        # Re-calculate Wilks scores on the filtered data
                        from models.wilks import calculate_wilks_scores
                        
                        # Convert to LazyFrame, calculate Wilks scores, then collect
                        temp_lf = filtered_df.lazy()
                        temp_lf = calculate_wilks_scores(temp_lf)
                        filtered_df = temp_lf.collect()
                        
                        logger.info(f"Recalculated Wilks scores, columns now: {filtered_df.columns[:10]}")
                    except Exception as e:
                        logger.error(f"Error calculating Wilks scores: {e}")

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
        filtered_df_cache.update({
            'key': filter_cache_key,
            'df': filtered_df,
            'row_count': filtered_df.height
        })
    else:
        # Use the cached filtered data
        logger.info(f"Using cached data for {triggered_id} - {filtered_df_cache['row_count']} rows")
        filtered_df = filtered_df_cache['df']
    
    # Calculate user total if all lifts are provided
    user_total = None
    if all(x is not None for x in [converted_squat, converted_bench, converted_deadlift]):
        user_total = converted_squat + converted_bench + converted_deadlift
    
    # Calculate user's weight class if bodyweight is provided
    user_weight_class = None
    if converted_bodyweight is not None and sex is not None and sex != 'All':
        user_weight_class = calculate_user_weight_class(converted_bodyweight, sex)
    
    # Determine which figures need to be updated based on the trigger
    # This optimized approach only updates the necessary figures
    figure_updates = get_metrics(triggered_id, need_full_update)

    # Ensure all Wilks plots update when bodyweight changes
    if triggered_id == 'bodyweight-input':
        for lift_type in figure_updates:
            figure_updates[lift_type]['wilks_histogram'] = True
            figure_updates[lift_type]['wilks_scatter'] = True

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
        user_wilks = None
        if sex and sex != 'All' and converted_bodyweight is not None and user_value is not None:
            logger.info(f"Calculating Wilks for {lift_type}: value={user_value}, bodyweight={converted_bodyweight}")
            user_wilks = get_wilks_value(user_value, converted_bodyweight, sex)
            logger.info(f"Calculated Wilks: {user_wilks}")
            try:
                user_wilks = get_wilks_value(user_value, converted_bodyweight, sex)
                logger.info(f"Calculated {lift_type} Wilks: {user_wilks} (value: {user_value}, bw: {converted_bodyweight})")
            except Exception as e:
                logger.error(f"Error calculating {lift_type} Wilks: {e}")
        else:
            logger.info(f"Cannot calculate Wilks: sex={sex}, bodyweight={converted_bodyweight}, value={user_value}")
            user_wilks = None        
        
        # Generate regular histogram if needed
        if figure_updates[lift_type]['histogram']:
            idx = figure_indices[lift_type]['histogram']
            try:
                start_time = time.time()
                results[idx] = create_histogram(filtered_df, equipment, lift_type, user_value)
                logger.info(f"Generated {lift_type} histogram in {time.time() - start_time:.2f}s")
            except Exception as e:
                results[idx] = empty_figure(f"Error generating {lift_type} histogram: {e}", e)
        
        # Generate Wilks histogram if needed
        idx = figure_indices[lift_type]['wilks_histogram']
        if sex != 'All' and converted_bodyweight is not None:
            try:
                # Explicitly check if data has Wilks columns
                wilks_col_name = f"{lift_type}Wilks"
                if wilks_col_name in filtered_df.columns:
                    results[idx] = create_histogram(filtered_df, equipment, lift_type, user_wilks, use_wilks=True)
                else:
                    results[idx] = empty_figure("Cannot create Wilks histogram: Wilks data not available", color="red")
            except Exception as e:
                results[idx] = empty_figure(f"Error generating {lift_type} Wilks histogram: {e}", e)
        elif sex != 'All':
            results[idx] = empty_figure("Please enter a bodyweight value to see the Wilks histogram")
        else:
            results[idx] = empty_figure("Wilks score requires selecting 'Male' or 'Female' in the Sex filter")
        
        # Generate scatter plot if needed
        idx = figure_indices[lift_type]['scatter']
        if converted_bodyweight is not None:
            
            try:
                start_time = time.time()
                results[idx] = create_scatter_plot(
                    filtered_df, equipment, lift_type, 
                    user_bodyweight=converted_bodyweight,
                    user_lift=user_value
                )
                logger.info(f"Generated {lift_type} scatter plot in {time.time() - start_time:.2f}s")
            except Exception as e:
                results[idx] = empty_figure(f"Error generating {lift_type} scatter plot: {e}", e)
        else:
            results[idx] = empty_figure("Please enter a bodyweight value to see the scatter plot")
        
        # Generate Wilks scatter plot if needed
        idx = figure_indices[lift_type]['wilks_scatter']
        if sex != 'All' and converted_bodyweight is not None:
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
                results[idx] = empty_figure(f"Error generating {lift_type} Wilks scatter plot: {e}", e)
        elif sex != 'All':
             results[idx] = empty_figure("Please enter a bodyweight value to see the Wilks scatter plot")
        else:
            results[idx] = empty_figure("Wilks score requires selecting 'Male' or 'Female' in the Sex filter")
    
    # Update last updated text if needed
    if need_full_update:
        # Get last updated date from data
        last_updated_file = config.get_data_path("last_updated_file")
        last_updated = "Unknown"
        if last_updated_file.exists():
            with last_updated_file.open(mode='r') as f:
                last_updated_str = f.read().strip()
                try:
                    last_updated = datetime.datetime.fromisoformat(last_updated_str).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
        
        sex_info = f" ({sex})" if sex != 'All' else ""
        equipment_info = f", Equipment: {', '.join(equipment)}" if equipment and len(equipment) > 0 else ""
        w_class = user_weight_class or weight_class
        weight_class_info = f", Weight Class: {w_class}" if w_class else ""
        
        row_count = filtered_df_cache['row_count']
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
     State('bodyweight-input', 'value'),
     State('units', 'value')]
)
def update_twitter_link(n_clicks: Optional[int], squat: Optional[float], bench: Optional[float],
                        deadlift: Optional[float], bodyweight: Optional[float], units: str) -> str:
    """Update the Twitter share link with the user's data."""
    if None in [squat, bench, deadlift, bodyweight]:
        return "#"
    
    # Format the units
    unit_text = "lbs" if units == 'imperial' else "kg"
    total = squat + bench + deadlift
    
    # Create share text
    share_text = (f"My powerlifting stats: Squat {squat}{unit_text}, "
                 f"Bench {bench}{unit_text}, Deadlift {deadlift}{unit_text}, "
                 f"Total {total}{unit_text} at {bodyweight}{unit_text} bodyweight. "
                 f"Check where you stand!")
    
    return f"https://twitter.com/intent/tweet?text={share_text}"

@app.callback(
    Output('share-facebook', 'href'),
    [Input('update-button', 'n_clicks')]
)
def update_facebook_link(n_clicks: Optional[int]) -> str:
    """Update the Facebook share link."""
    return "https://www.facebook.com/sharer/sharer.php?u=https://powerlifting-visualizer.example.com"

@app.callback(
    Output('share-instagram', 'href'),
    [Input('update-button', 'n_clicks')]
)
def update_instagram_link(n_clicks: Optional[int]) -> str:
    """Update the Instagram share link (placeholder)."""
    return "#"

# ---------------- Main Entry Point ----------------
if __name__ == '__main__':
    app.run(debug=DEBUG_MODE)
