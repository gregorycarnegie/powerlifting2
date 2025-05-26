import hashlib
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import dash
import plotly.graph_objects as go
import polars as pl

from services.cache_service import cache_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    """
    Register a client-side callback to validate numeric input for the given input_id.

    Args:
        dash_app: Dash application instance
        input_id: ID of the input element to validate
    """
    dash_app.clientside_callback(
        numeric_validation_js,
        dash.Output(input_id, 'value'),
        [dash.Input(input_id, 'value')],
        [dash.State(input_id, 'id')]
    )

def get_dataframe_hash(df: pl.DataFrame) -> str:
    """Generate a hash for a DataFrame."""
    if df is None:
        return "none"

    if not (hasattr(df, 'height') and hasattr(df, 'columns')):
        return f"obj_{id(df)}"

    try:
        sample_values = _extract_sample_values(df)
        column_info = f"{df.height}_{len(df.columns)}_{'-'.join(df.columns[:5])}"
        df_data = f"{column_info}_{'_'.join(sample_values)}"
        return hashlib.blake2b(df_data.encode(), digest_size=8).hexdigest()
    except Exception as e:
        logger.error(f"Error generating DataFrame hash: {e}")
        return f"df_{hash(tuple(df.columns)) if hasattr(df, 'columns') else id(df)}"

def _extract_sample_values(df: pl.DataFrame) -> list[str]:
    """Extract sample values from the DataFrame for hash generation."""
    sample_vals = []
    if df.height == 0:
        return sample_vals

    n_cols = min(3, len(df.columns))
    n_rows = min(3, df.height)

    for col in df.columns[:n_cols]:
        for i in range(n_rows):
            try:
                sample_vals.append(str(df[col][i]))
            except Exception as e:
                logger.error(f"Error getting sample value: {e}")

    return sample_vals[:10]

def format_arg_for_key(arg: Any) -> str:
    """Format a single argument for inclusion in a cache key."""
    if arg is None:
        return "none"
    elif isinstance(arg, int | float | bool):
        return str(arg)
    elif isinstance(arg, str):
        return arg[:10]  # First 10 chars of strings
    else:
        return f"obj_{id(arg)}"

def format_args_for_key(args: tuple) -> str:
    """Format positional arguments for a cache key."""
    return "_".join(format_arg_for_key(arg) for arg in args)

def format_kwargs_for_key(kwargs: dict) -> str:
    """Format keyword arguments for a cache key."""
    formatted_items = []

    for k, v in sorted(kwargs.items()):
        if v is None:
            formatted_items.append(f"{k}_none")
        elif isinstance(v, int | float | bool):
            formatted_items.append(f"{k}_{v}")
        elif isinstance(v, str):
            formatted_items.append(f"{k}_{v[:10]}")
        else:
            formatted_items.append(f"{k}_obj_{id(v)}")

    return "_".join(formatted_items)

def advanced_cache_key(
        func_name: str,
        df: pl.DataFrame,
        *args,
        **kwargs
) -> str:
    """
    Generate a sophisticated cache key for figures.

    Args:
        func_name: Name of the function generating the figure
        df: DataFrame being visualized
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        Hash string to use as a cache key
    """
    df_hash = get_dataframe_hash(df)
    args_str = format_args_for_key(args)
    kwargs_str = format_kwargs_for_key(kwargs)

    # Combine all parts
    key_parts = [func_name, df_hash, args_str, kwargs_str]
    combined_key = "_".join(key_parts)

    # Use a fast hash function
    return hashlib.blake2b(combined_key.encode(), digest_size=8).hexdigest()

def advanced_cached_figure(func: Callable) -> Callable:
    """
    Enhanced decorator for caching figures with comprehensive metrics.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Skip caching for dash.no_update or None values
        if args and any(arg is dash.no_update or arg is None for arg in args):
            return dash.no_update
        if kwargs and any(v is dash.no_update or v is None for v in kwargs.values()):
            return dash.no_update

        # Get dataframe from args (typically first arg)
        df = args[0] if args else None

        # Generate a sophisticated cache key
        cache_key = advanced_cache_key(func.__name__, df, *args[1:], **kwargs)

        # Try to get from the cache
        cached_result = cache_service.get(cache_key)
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
            cache_service.put(cache_key, result)

            return result
        except Exception as e:
            logger.error(f"Error generating figure {func.__name__}: {e}")
            # Return an empty figure with an error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating figure: {str(e)}",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            return fig

    return wrapper
