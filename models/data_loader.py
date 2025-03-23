import datetime
import io
import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

from models.wilks import calculate_ipf_weight_class, calculate_wilks_scores
from services.config_service import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = config.get_data_path()
PARQUET_FILE = config.get_data_path("parquet_file")
LAST_UPDATED_FILE = config.get_data_path("last_updated_file")
CSV_URL = config.get("data", "csv_url")
CSV_BACKUP = config.get_data_path("csv_backup")

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

def check_for_updates() -> bool:
    """
    Check if we need to update the dataset.
    
    Returns:
        True if update is needed, False otherwise
    """
    update_interval_days = config.get("data", "update_interval_days") or 1
    
    if not PARQUET_FILE.exists():
        return True
    if not LAST_UPDATED_FILE.exists():
        return True
    
    with LAST_UPDATED_FILE.open(mode='r') as f:
        last_updated_str = f.read().strip()
    last_updated = datetime.datetime.fromisoformat(last_updated_str)
    now = datetime.datetime.now()
    
    return (now - last_updated).days >= update_interval_days

def download_and_process_data() -> pl.LazyFrame:
    """
    Download and process OpenPowerlifting data using an explicit schema.
    
    Returns:
        Processed LazyFrame
    """
    logger.info("Downloading latest OpenPowerlifting data...")
    response = requests.get(CSV_URL)
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(f"Failed to download data: {response.status_code}")

    # Extract the CSV file from the ZIP archive
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        return zip_to_lazyframe(z)

def zip_to_lazyframe(z: zipfile.ZipFile) -> pl.LazyFrame:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the ZIP archive")
    csv_file = csv_files[0]
    logger.info(f"Extracting {csv_file}...")
    z.extract(csv_file, DATA_DIR)
    csv_path = DATA_DIR / csv_file

    # Process the extracted data
    return process_powerlifting_data(csv_path)

def process_powerlifting_data(csv_path: Path) -> pl.LazyFrame:
    """
    Process powerlifting data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed LazyFrame
    """
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
            (pl.col('Best3SquatKg').gt(0)) |
            (pl.col('Best3BenchKg').gt(0)) |
            (pl.col('Best3DeadliftKg').gt(0)) |
            (pl.col('TotalKg').gt(0))
        )
    )
    
    # Calculate weight class
    logger.info("Calculating weight classes...")
    base_lf = base_lf.with_columns([
        calculate_ipf_weight_class(
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
            'filter': pl.col('Best3SquatKg').gt(0)
        },
        {
            'name': 'Bench',
            'value_col': 'Best3BenchKg',
            'filter': pl.col('Best3BenchKg').gt(0)
        },
        {
            'name': 'Deadlift',
            'value_col': 'Best3DeadliftKg',
            'filter': pl.col('Best3DeadliftKg').gt(0)
        },
        {
            'name': 'Total',
            'value_col': 'TotalKg',
            'filter': pl.col('TotalKg').gt(0)
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
    result_lf = calculate_wilks_scores(result_lf)
    
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
            collected_df.write_csv(CSV_BACKUP)
            logger.info(f"Saved data as CSV to {CSV_BACKUP}")
    
    with LAST_UPDATED_FILE.open(mode='w') as f:
        f.write(datetime.datetime.now().isoformat())
    
    logger.info("Data update complete!")
    return result_lf

def load_data() -> pl.LazyFrame:
    """
    Load the data, downloading it first if necessary.
    
    Returns:
        Processed LazyFrame
    """
    if check_for_updates():
        return download_and_process_data()
    logger.info("Loading cached data...")
    try:
        return pl.scan_parquet(PARQUET_FILE)
    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")

        # Check if we have a CSV backup
        if CSV_BACKUP.exists():
            logger.info(f"Trying to load from CSV backup: {CSV_BACKUP}")
            try:
                return pl.scan_csv(CSV_BACKUP)
            except Exception as csv_e:
                logger.error(f"Error loading CSV backup: {csv_e}")

        # If we get here, we need to re-download and process the data
        logger.info("Re-downloading and processing data...")
        return download_and_process_data()
