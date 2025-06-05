import logging
from functools import cache
from typing import Literal, TypeVar

import polars as pl

from services.config_service import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load coefficients from config
MALE_WILKS_COEFFICIENTS = tuple(config.get("visualization", "wilks_coefficients_male"))
FEMALE_WILKS_COEFFICIENTS = tuple(config.get("visualization", "wilks_coefficients_female"))

# Define weight classes
WEIGHT_CLASSES: dict[str, tuple[tuple[float, str]]] = {
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

# Junior weight classes
JUNIOR_WEIGHT_CLASSES = {
    "M": (53.0, "53kg"),
    "F": (43.0, "43kg")
}

T = TypeVar("T")


def wilks_polynomial(coefficients: tuple[float, ...], bodyweight: T) -> T:
    """
    Calculate Wilks score using the provided coefficients and bodyweight.

    Args:
        coefficients: Tuple of coefficients for the Wilks formula
        bodyweight: Bodyweight in kg

    Returns:
        Wilks score
    """
    return 600 / sum(x * bodyweight ** y for x, y in zip(coefficients, range(6)))

def calculate_wilks_scores(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate Wilks scores for different lifts.

    Args:
        df: Input LazyFrame containing lifting data

    Returns:
        LazyFrame with added Wilks score columns
    """

    # Define a function to create the Wilks formula expression
    def wilks_formula(lift_col: str, bw_col: str, sex_col: pl.Expr) -> pl.Expr:
        return (
            pl.when((pl.col(lift_col) > 0) & (pl.col(bw_col) > 0))
            .then(
                pl.when(sex_col == 'M')
                .then(
                    pl.col(lift_col) * wilks_polynomial(MALE_WILKS_COEFFICIENTS, pl.col(bw_col))
                )
                .otherwise(
                    pl.col(lift_col) * wilks_polynomial(FEMALE_WILKS_COEFFICIENTS, pl.col(bw_col))
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
def get_wilks_value(
    total: float | None,
    bodyweight: float | None,
    sex: Literal['M', 'F', 'All'] | None
) -> float | None:
    """
    Calculate the Wilks score based on total, bodyweight, and sex.

    Args:
        total: Lift total in kg
        bodyweight: Lifter's bodyweight in kg
        sex: Lifter's sex ('M', 'F', or 'All')

    Returns:
        Calculated Wilks score or None if input is invalid
    """
    if total is None or bodyweight is None or sex is None or sex == 'All':
        return None

    if sex == 'M':
        return total * wilks_polynomial(MALE_WILKS_COEFFICIENTS, bodyweight)
    elif sex == 'F':
        return total * wilks_polynomial(FEMALE_WILKS_COEFFICIENTS, bodyweight)

    return None

def calculate_ipf_weight_class(
        bodyweight_col: pl.Expr,
        sex_col: pl.Expr,
        age_col: pl.Expr | None = None
) -> pl.Expr:
    """
    Calculate weight class using Polars expressions.

    Args:
        bodyweight_col: Column with bodyweight values
        sex_col: Column with sex values ('M' or 'F')
        age_col: Optional column with age values

    Returns:
        Polars expression that can be used in with_columns()
    """
    # Check if junior
    is_junior = pl.when(age_col.is_not_null() & (age_col.lt(23)) & (age_col.gt(0))).then(True).otherwise(False) if isinstance(age_col, pl.Expr) else pl.lit(False)

    # Create the weight class expression using when>then>otherwise chain
    result = pl.lit(None).cast(pl.Utf8)

    # Add junior classes first
    for sex, (limit, class_name) in JUNIOR_WEIGHT_CLASSES.items():
        result = (
            pl.when((sex_col.eq(sex)) & is_junior & (bodyweight_col.le(limit)))
            .then(pl.lit(class_name))
            .otherwise(result)
        )

    # Process weight classes for each sex
    for sx in ['M', 'F']:
        prev_limit = 0.0
        for limit, class_name in WEIGHT_CLASSES[sx]:
            result = (
                pl.when((sex_col.eq(sx)) & (bodyweight_col.gt(prev_limit)) & (bodyweight_col.le(limit)))
                .then(pl.lit(class_name))
                .otherwise(result)
            )
            prev_limit = limit

    return result

def calculate_user_weight_class(bodyweight: float, sex: Literal['M', 'F', 'All']) -> str:
    """
    Calculate the user's weight class based on bodyweight and sex.

    Args:
        bodyweight: User's bodyweight in kg
        sex: User's sex ('M', 'F', or 'All')

    Returns:
        Weight class as a string
    """
    if sex == 'All':
        return "Unknown"

    return next(
        (
            weight_class
            for weight_limit, weight_class in WEIGHT_CLASSES.get(sex, [])
            if bodyweight <= weight_limit
        ),
        "Unknown",
    )

def get_weight_class_options(sex: Literal['M', 'F', 'All'] = 'M') -> list[dict[str, str]]:
    """
    Get the weight class options based on sex.

    Args:
        sex: 'M', 'F', or 'All'

    Returns:
        List of dictionaries with label/value pairs for dropdown
    """
    if sex == 'M':
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
    elif sex == 'F':
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
    else:  # 'All'
        return [{'label': 'All Weight Classes', 'value': 'all'}]

def get_lift_columns(lift: str, bdy: bool = False) -> tuple[str, str, str | None]:
    """
    Get the column names for the given lift.

    Args:
        lift: Lift type ('Squat', 'Bench', 'Deadlift', or 'Total')
        bdy: Whether to include bodyweight column

    Returns:
        Tuple of (lift_column, wilks_column, bodyweight_column)
    """
    if lift in {'Squat', 'Bench', 'Deadlift', 'Total'}:
        return (
            'TotalKg' if lift == 'Total' else f'Best3{lift}Kg',
            f'{lift}Wilks',
            f'{lift}BodyweightKg' if bdy else None
        )
    else:
        raise ValueError(f"Invalid lift: {lift}")
