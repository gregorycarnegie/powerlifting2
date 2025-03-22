import logging
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl

from models.wilks import get_lift_columns
from services.config_service import config
from utils.dash_helpers import advanced_cached_figure
from utils.filters import get_specific_filter, sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load sample size from config
DEFAULT_SAMPLE_SIZE = config.get("visualization", "default_sample_size") or 10000
BODYWEIGHT_TOLERANCE = config.get("visualization", "bodyweight_tolerance") or 5

@advanced_cached_figure
def create_histogram(df: pl.DataFrame, equipment: List[str], lift: str,
                     user_value: Optional[float] = None, use_wilks: bool = False,
                     sample_size: int = DEFAULT_SAMPLE_SIZE) -> go.Figure:
    """
    Create a histogram for the specified lift with improved performance.
    
    Args:
        df: Input DataFrame
        equipment: List of equipment types to include
        lift: Lift type
        user_value: User's lift value to highlight
        use_wilks: Whether to use Wilks score
        sample_size: Maximum sample size for visualization
        
    Returns:
        Plotly Figure object
    """
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
        column, wilks_column, _ = get_lift_columns(lift)
    except ValueError as e:
        logger.error(f"Unknown lift type: {e}")
        return go.Figure()

    # Check if the Wilks column exists when requested
    if use_wilks and wilks_column not in df.columns:
        logger.warning(f"{wilks_column} not found in dataframe. Falling back to regular values.")
        use_wilks = False
    
    # Fast filtering with expression
    plot_df = get_specific_filter(df, column, lift, equipment)
    
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
    plot_df = sample_data(plot_df, lift, sample_size, create_histogram.__name__)

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

@advanced_cached_figure
def create_scatter_plot(df: pl.DataFrame, equipment: List[str], lift: str, use_wilks: bool = False,
                        user_bodyweight: Optional[float] = None, user_lift: Optional[float] = None,
                        sample_size: int = DEFAULT_SAMPLE_SIZE) -> go.Figure:
    """
    Create a scatter plot of the lift vs. bodyweight with improved performance.
    
    Args:
        df: Input DataFrame
        equipment: List of equipment types to include
        lift: Lift type
        use_wilks: Whether to use Wilks score
        user_bodyweight: User's bodyweight to highlight
        user_lift: User's lift value to highlight
        sample_size: Maximum sample size for visualization
        
    Returns:
        Plotly Figure object
    """
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
        column, wilks_column, bodyweight_column = get_lift_columns(lift, bdy=True)
    except ValueError as e:
        logger.error(f"Unknown lift type: {e}")
        return go.Figure()
    
    # Check if the Wilks column exists when requested
    if use_wilks and wilks_column not in df.columns:
        logger.warning(f"{wilks_column} not found in dataframe. Falling back to regular values.")
        use_wilks = False
    
    # Filter for rows with positive lift and bodyweight values
    plot_df = get_specific_filter(df, column, lift, equipment, bodyweight_column)
    
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
    plot_df = sample_data(plot_df, lift, sample_size, create_scatter_plot.__name__)

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
            weight_range = BODYWEIGHT_TOLERANCE  # kg
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

def empty_figure(text: str, error: Optional[Exception]=None, *, color: Optional[str]=None) -> go.Figure:
    """Create an empty figure with a message."""
    empty_fig = go.Figure()
    if error:
        logger.error(text)
        empty_fig.add_annotation(text=text, showarrow=False, font=dict(size=14, color=color or "red"))
    else:
        empty_fig.add_annotation(text=text, showarrow=False, font=dict(size=14, color=color or "blue"))
    return empty_fig
