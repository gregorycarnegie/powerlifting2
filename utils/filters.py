import logging
from functools import cache
from typing import Literal

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def filter_data(lf: pl.LazyFrame, sex: Literal['M', 'F', 'All'] | None = None,
                equipment: list[str] | None = None, weight_class: str | None = None) -> pl.LazyFrame:
    """
    Filter the LazyFrame based on user selections.

    Args:
        lf: Input LazyFrame
        sex: Sex filter ('M', 'F', or 'All')
        equipment: List of equipment types to include
        weight_class: Weight class filter

    Returns:
        Filtered LazyFrame
    """
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

def get_specific_filter(df: pl.DataFrame, column: str, lift: str, equipment: list[str],
                        bodyweight_column: str | None = None) -> pl.DataFrame:
    """
    Apply lift-specific filtering to a DataFrame.

    Args:
        df: Input DataFrame
        column: Column to filter on
        lift: Lift type
        equipment: List of equipment types to include
        bodyweight_column: Optional bodyweight column to filter on

    Returns:
        Filtered DataFrame
    """
    if equipment:
        result = df.filter(
            (pl.col(column) > 0) &
            (pl.col(f'{lift}Equipment').is_in(equipment))
        )
    else:
        result = df.filter(pl.col(column) > 0)

    if bodyweight_column:
        return result.filter(pl.col(bodyweight_column) > 0)

    return result

def sample_data(df: pl.DataFrame, lift: str, sample_size: int, func_name: str) -> pl.DataFrame:
    """
    Sample the data for visualizations to improve performance.

    Args:
        df: Input DataFrame
        lift: Lift type
        sample_size: Target sample size
        func_name: Name of the calling function (for logging)

    Returns:
        Sampled DataFrame
    """
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

        logger.info(f"{func_name}: Sampled {df.height} rows for {lift} visualization")

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

    # Special handling for bodyweight changes
    bodyweight_changed = triggered_id == 'bodyweight-input'
    if bodyweight_changed:
        common_inputs.add('bodyweight-input')

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
    figure_updates = {lift_name: {} for lift_name in lift_mapping}
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
