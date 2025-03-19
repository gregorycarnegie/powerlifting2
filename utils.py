import hashlib
import logging
import time
from collections.abc import Callable
from functools import wraps, cache
from typing import Optional, Literal, Union

import dash
import plotly.graph_objects as go
import polars as pl
from dash.dependencies import Input, Output, State

from figure_cache import cache_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Male Wilks coefficients
MWC = -216.0475144, 16.2606339, -0.002388645, -0.00113732, 7.01863e-06, -1.291e-08

# Female Wilks coefficients
FWC = 594.31747775582, -27.23842536447, 0.82112226871, -0.00930733913, 4.731582e-05, -9.054e-08

# Define weight classes

weight_classes = {
    'M': (
        (59.0, "59kg"),
        (66.0, "66kg"),
        (74.0, "74kg"),
        (83.0, "83kg"),
        (93.0, "93kg"),
        (105.0, "105kg"),
        (120.0, "120kg"),
        (float('inf'), "120kg+")
    ),
    'F': (
        (47.0, "47kg"),
        (52.0, "52kg"),
        (57.0, "57kg"),
        (63.0, "63kg"),
        (69.0, "69kg"),
        (76.0, "76kg"),
        (84.0, "84kg"),
        (float('inf'), "84kg+")
    )
}

junior_classes = {
    "M": (53.0, "53kg"),
    "F": (43.0, "43kg")
}

# Define a reusable client-side callback for numeric validation
numeric_validation_js = """
function(value, id) {
    // If the input is null or undefined, return it as is
    if (value === null || value === undefined) {
        return value;
    }
    
    // Convert to string to handle all input types
    value = value.toString();
    
    // Remove any non-numeric characters except decimal point
    // Keep only digits and at most one decimal point
    let hasDecimal = false;
    let result = '';
    
    for (let i = 0; i < value.length; i++) {
        const char = value[i];
        
        // Allow digits
        if (char >= '0' && char <= '9') {
            result += char;
        }
        // Allow one decimal point
        else if (char === '.' && !hasDecimal) {
            result += char;
            hasDecimal = true;
        }
        // Skip any other characters
    }
    
    // If result is empty, return null
    if (result === '' || result === '.') {
        return null;
    }
    
    // Convert back to number
    return parseFloat(result);
}
"""

def register_numeric_validation(dash_app: dash.Dash, input_id: str) -> None:
    """Register a client-side callback to validate numeric input for the given input_id"""
    dash_app.clientside_callback(
        numeric_validation_js,
        Output(input_id, 'value'),
        [Input(input_id, 'value')],
        [State(input_id, 'id')]
    )

def calculate_wilks_scores(df: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate Wilks scores for different lifts."""
    # Define coefficients for Wilks calculation
    # Male coefficients
    a_m, b_m, c_m, d_m, e_m, f_m = MWC
    
    # Female coefficients
    a_f, b_f, c_f, d_f, e_f, f_f = FWC
    
    # Define a function to create the Wilks formula expression
    def wilks_formula(lift_col, bw_col, sex_col):
        return (
            pl.when((pl.col(lift_col) > 0) & (pl.col(bw_col) > 0))
            .then(
                pl.when(sex_col == 'M')
                .then(
                    pl.col(lift_col) * 500 / (
                        a_m + 
                        b_m * pl.col(bw_col) + 
                        c_m * pl.col(bw_col)**2 + 
                        d_m * pl.col(bw_col)**3 + 
                        e_m * pl.col(bw_col)**4 + 
                        f_m * pl.col(bw_col)**5
                    )
                )
                .otherwise(
                    pl.col(lift_col) * 500 / (
                        a_f + 
                        b_f * pl.col(bw_col) + 
                        c_f * pl.col(bw_col)**2 + 
                        d_f * pl.col(bw_col)**3 + 
                        e_f * pl.col(bw_col)**4 + 
                        f_f * pl.col(bw_col)**5
                    )
                )
            )
            .otherwise(0)
        )
    
    # Calculate wilks for each lift type using the correct column names
    wilks_calculations = [
        wilks_formula('Best3SquatKg', 'SquatBodyweightKg', pl.col('Sex')).alias('SquatWilks'),
        wilks_formula('Best3BenchKg', 'BenchBodyweightKg', pl.col('Sex')).alias('BenchWilks'),
        wilks_formula('Best3DeadliftKg', 'DeadliftBodyweightKg', pl.col('Sex')).alias('DeadliftWilks'),
        wilks_formula('TotalKg', 'TotalBodyweightKg', pl.col('Sex')).alias('TotalWilks')
    ]
    
    return df.with_columns(wilks_calculations)

@cache
def get_wilks_value(total: Optional[float], bodyweight: Optional[float], sex: Optional[Literal['M', 'F', 'All']]) -> Optional[float]:
    """Calculate the Wilks score based on total, bodyweight, and sex"""
    if total is None or bodyweight is None or sex is None:
        return None
    match sex:
        case 'M':
            # Male Wilks coefficients
            polynomial = sum(x * bodyweight ** y for x, y in zip(MWC, range(6)))
            return total * 500 / polynomial
        case 'F':
            # Female Wilks coefficients
            polynomial = sum(x * bodyweight ** y for x, y in zip(FWC, range(6)))
            return total * 500 / polynomial
        case 'All':
            return None

def advanced_cache_key(func_name: str, df: pl.DataFrame, *args, **kwargs) -> str:
    """Generate a sophisticated cache key for figures."""
    # Create a DataFrame fingerprint
    if df is None:
        df_hash = "none"
    elif hasattr(df, 'height') and hasattr(df, 'columns'):
        # This handles both Polars DataFrames and any DataFrame-like object
        try:
            # Try to include sample values for more uniqueness
            sample_vals = []
            if df.height > 0:
                n_cols = min(3, len(df.columns))
                n_rows = min(3, df.height)
                for col in df.columns[:n_cols]:
                    for i in range(n_rows):
                        try:
                            sample_vals.append(str(df[col][i]))
                        except Exception as e:
                            logger.error(f"Error getting sample value: {e}")
                            pass
            
            # Combine structural info with sample values
            df_data = f"{df.height}_{len(df.columns)}_{'-'.join(df.columns[:5])}_{'_'.join(sample_vals[:10])}"
            df_hash = hashlib.blake2b(df_data.encode(), digest_size=8).hexdigest()
        except Exception as e:
            logger.error(f"Error generating DataFrame hash: {e}")
            # Fallback for error cases
            df_hash = f"df_{hash(tuple(df.columns)) if hasattr(df, 'columns') else id(df)}"
    else:
        # For other types
        df_hash = f"obj_{id(df)}"
    
    # Process args and kwargs
    args_str = []
    for arg in args:
        if arg is None:
            args_str.append("none")
        elif isinstance(arg, (int, float, bool)):
            args_str.append(str(arg))
        elif isinstance(arg, str):
            args_str.append(arg[:10])  # First 10 chars of strings
        else:
            args_str.append(f"obj_{id(arg)}")
    
    kwargs_str = []
    for k, v in sorted(kwargs.items()):
        if v is None:
            kwargs_str.append(f"{k}_none")
        elif isinstance(v, (int, float, bool)):
            kwargs_str.append(f"{k}_{v}")
        elif isinstance(v, str):
            kwargs_str.append(f"{k}_{v[:10]}")  # First 10 chars of strings
        else:
            kwargs_str.append(f"{k}_obj_{id(v)}")
    
    # Combine all parts
    key_parts = [func_name, df_hash, "_".join(args_str), "_".join(kwargs_str)]
    combined_key = "_".join(key_parts)
    
    # Use a fast hash function
    return hashlib.blake2b(combined_key.encode(), digest_size=8).hexdigest()

def advanced_cached_figure(func: Callable):
    """Enhanced decorator for caching figures with comprehensive metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Skip caching for dash.no_update or None values
        if args and any(arg is dash.no_update or arg is None for arg in args):
            return dash.no_update
        if kwargs and any(v is dash.no_update or v is None for v in kwargs.values()):
            return dash.no_update
        
        # Get dataframe from args (typically first arg)
        df = args[0] if args else None
        
        # Generate sophisticated cache key
        cache_key = advanced_cache_key(func.__name__, df, *args[1:], **kwargs)
        
        # Try to get from cache
        cached_result = cache_system.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for {func.__name__}")
            return cached_result
        
        # Not in cache, generate the figure
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} ran in {execution_time:.3f}s")
            
            # Store in cache
            cache_system.put(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error generating figure {func.__name__}: {e}")
            # Return an empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating figure: {str(e)}",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            return fig
    
    return wrapper

def filter_data(lf: pl.LazyFrame, sex: Optional[Literal['M', 'F', 'All']]=None, equipment:Optional[list[str]]=None, weight_class: Optional[str]=None) -> pl.LazyFrame:
    """Filter the LazyFrame based on user selections."""
    if sex and sex != 'All':
        logger.info(f"Filtering by sex: {sex}")
        lf = lf.filter(pl.col('Sex') == sex)
    
    # Equipment filtering needs to check each lift type's equipment
    if equipment and len(equipment) > 0:
        logger.info(f"Filtering by equipment: {equipment}")
        equipment_filter = (
            pl.col('SquatEquipment').is_in(equipment) |
            pl.col('BenchEquipment').is_in(equipment) |
            pl.col('DeadliftEquipment').is_in(equipment) |
            pl.col('TotalEquipment').is_in(equipment)
        )
        lf = lf.filter(equipment_filter)
    
    # Weight class filtering
    if weight_class and weight_class != 'all':
        logger.info(f"Filtering by weight class: {weight_class}")
        weight_class_filter = (
            (pl.col('SquatWeightClassKg') == weight_class) |
            (pl.col('BenchWeightClassKg') == weight_class) |
            (pl.col('DeadliftWeightClassKg') == weight_class) |
            (pl.col('TotalWeightClassKg') == weight_class)
        )
        lf = lf.filter(weight_class_filter)
    
    return lf

def get_weight_class_options(sex: Literal['M', 'F', 'All']='M') -> Optional[Union[list, list[dict[str, str]]]]:
    """Get the weight class options based on sex"""
    match sex:
        case 'M':
            return [
                {'label': 'All Weight Classes', 'value': 'all'},
                {'label': '53kg (Junior only)', 'value': '53kg'},
                {'label': '59kg', 'value': '59kg'},
                {'label': '66kg', 'value': '66kg'},
                {'label': '74kg', 'value': '74kg'},
                {'label': '83kg', 'value': '83kg'},
                {'label': '93kg', 'value': '93kg'},
                {'label': '105kg', 'value': '105kg'},
                {'label': '120kg', 'value': '120kg'},
                {'label': '120kg+', 'value': '120kg+'}
            ]
        case 'F':
            return [
                {'label': 'All Weight Classes', 'value': 'all'},
                {'label': '43kg (Junior only)', 'value': '43kg'},
                {'label': '47kg', 'value': '47kg'},
                {'label': '52kg', 'value': '52kg'},
                {'label': '57kg', 'value': '57kg'},
                {'label': '63kg', 'value': '63kg'},
                {'label': '69kg', 'value': '69kg'},
                {'label': '76kg', 'value': '76kg'},
                {'label': '84kg', 'value': '84kg'},
                {'label': '84kg+', 'value': '84kg+'}
            ]
        case 'All':
            return []

def calculate_ipf_weight_class(bodyweight_col: pl.Expr, sex_col: pl.Expr, age_col: pl.Expr = None) -> pl.Expr:
    """
    Calculate weight class using Polars expressions.
    Returns a Polars expression that can be used in with_columns().
    """
    # Define weight classes as tuples of (upper_limit, class_name)
    
    # Check if junior
    is_junior = pl.when(age_col.is_not_null() & (age_col < 23) & (age_col > 0)).then(True).otherwise(False)
    
    # Create the weight class expression using when-then-otherwise chain
    result = pl.lit(None).cast(pl.Utf8)
    
    # Add junior classes first
    for sex, (limit, class_name) in junior_classes.items():
        result = (
            pl.when((sex_col == sex) & is_junior & (bodyweight_col <= limit))
            .then(pl.lit(class_name))
            .otherwise(result)
        )
    
    # Process male weight classes
    for sx in ['M', 'F']:
        prev_limit = 0.0
        for limit, class_name in weight_classes[sx]:
            result = (
                pl.when((sex_col == sx) & (bodyweight_col > prev_limit) & (bodyweight_col <= limit))
                .then(pl.lit(class_name))
                .otherwise(result)
            )
            prev_limit = limit
    
    return result

def calculate_user_weight_class(bodyweight: float, sex: Literal['M', 'F', 'All']) -> str:
    """Calculate the user's weight class based on bodyweight and sex."""
    for weight_limit, weight_class in weight_classes.get(sex, []):
        if bodyweight <= weight_limit:
            return weight_class
    
    return "Invalid sex parameter"

def get_specific_filter(df: pl.DataFrame, column: str, lift: str, equipment: list[str], bodyweight_column: Optional[str]=None) -> pl.DataFrame:
    if equipment and len(equipment) > 0:
        result = df.filter(
            (pl.col(column) > 0) &
            (pl.col(f'{lift}Equipment').is_in(equipment))
        )
    else:
        result = df.filter(pl.col(column) > 0)
    if bodyweight_column:
        return result.filter(pl.col(bodyweight_column) > 0)
    return result

def get_lift_columns(lift: str, bdy: bool=False) -> tuple[str, str, Optional[str]]:
    """Get the column names for the given lift."""
    if lift in ['Squat', 'Bench', 'Deadlift', 'Total']:
        return (
            'TotalKg' if lift == 'Total' else f'Best3{lift}Kg',
            f'{lift}Wilks',
            f'{lift}BodyweightKg' if bdy else None
        )
    else:
        raise ValueError(f"Invalid lift: {lift}")

def sample_data(df: pl.DataFrame, lift: str, sample_size: int, func: Callable) -> pl.DataFrame:
    """Sample the data for scatter plots."""
    # Efficient sampling strategy
    if df.height > sample_size:
        # Get unique sexes
        unique_sexes: list[str] = df['Sex'].unique().to_list()
        
        # For stratified sampling by sex
        if len(unique_sexes) > 1:
            # Get counts for each sex
            sex_counts = {sex: df.filter(pl.col('Sex') == sex).height for sex in unique_sexes}
            total_count = sum(sex_counts.values())
            
            # Sample proportionally
            sampled_dfs = []
            for sex, count in sex_counts.items():
                if count > 0:
                    # Calculate proportional sample size
                    sex_sample_size = max(1, min(int(sample_size * count / total_count), count))
                    sex_df = df.filter(pl.col('Sex') == sex)
                    
                    # Use random sampling
                    sampled_dfs.append(sex_df.sample(n=sex_sample_size))
            
            # Combine samples
            df = pl.concat(sampled_dfs)
        else:
            # Simple random sampling for single sex
            df = df.sample(n=min(sample_size, df.height))
        
        logger.info(f"{func.__name__}: Sampled {df.height} rows for {lift} visualization")
    
    return df

@cache
def get_metrics(triggered_id: str, need_full_update: bool) -> dict[str, dict[str, bool]]:
    """
    Determine which metrics need to be updated based on the triggered component and update flag.
    
    Args:
        triggered_id: ID of the component that triggered the callback
        need_full_update: Flag indicating if a full update is needed
        
    Returns:
        Dictionary mapping exercises to chart types with boolean flags indicating if update is needed
    """
    # Common input controls that affect all charts
    common_inputs = {'update-button', 'sex-filter', 'equipment-filter', 'weight-class-dropdown', 'units'}
    
    # Map display names to input IDs
    lift_mapping = {
        'Squat': 'squat',
        'Bench': 'bench',
        'Deadlift': 'deadlift'
    }
    
    # Define chart types and their additional required inputs beyond the common ones
    chart_additional_inputs = {
        'histogram': set(),  # No additional inputs beyond lift-specific and common inputs
        'wilks_histogram': {'bodyweight-input'},
        'scatter': {'bodyweight-input'},
        'wilks_scatter': {'bodyweight-input'}
    }
    
    # Initialize result dictionary
    figure_updates = {lift_name: {} for lift_name in lift_mapping.keys()}
    figure_updates['Total'] = {}
    
    # Fill in individual lift metrics
    for lift_name, lift_id in lift_mapping.items():
        for chart_name, additional_inputs in chart_additional_inputs.items():
            input_set = common_inputs | {f'{lift_id}-input'} | additional_inputs
            figure_updates[lift_name][chart_name] = triggered_id in input_set or need_full_update
    
    # Handle Total metrics (affected by any lift input)
    all_lift_inputs = {f'{lift_id}-input' for lift_id in lift_mapping.values()}
    
    for chart_name, additional_inputs in chart_additional_inputs.items():
        total_inputs = common_inputs | all_lift_inputs | additional_inputs
        figure_updates['Total'][chart_name] = triggered_id in total_inputs or need_full_update
    
    return figure_updates
