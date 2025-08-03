# Powerlifting Data Visualization App

This application visualizes powerlifting data from OpenPowerlifting, allowing users to compare their lifts with data from competitive powerlifters around the world.

## Features

- Automatically downloads and processes the latest OpenPowerlifting dataset
- Interactive histograms for squat, bench press, deadlift, and total with "you-are-here" indicators
- Scatter plots comparing lifts to bodyweight, color-coded by sex
- Wilks score calculation and visualization to compare lifters across weight classes
- Filtering by sex, equipment type, and weight class
- Social media sharing functionality
- Responsive design that works on desktop and mobile devices
- Automatic data refreshing to keep information current

## Installation

1. Clone this repository
2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:8050`

## Data Source

This application uses data from the [OpenPowerlifting project](https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html), which provides comprehensive powerlifting competition results in CSV format. The app automatically downloads the latest data when first run and checks for updates daily.

## Usage

1. Enter your personal lift numbers (squat, bench, deadlift) and bodyweight
2. Select your preferred filters (sex, equipment type, weight class)
3. Click "Update Visualizations" to see how your lifts compare
4. Navigate between the different lift tabs to view detailed comparisons
5. Share your results on social media using the share buttons

## Technical Details

The app uses:

- Dash and Plotly for interactive visualizations
- Pandas and PyArrow for efficient data processing
- Bootstrap for responsive layout
- Wilks formula to normalize lifts across different bodyweights

The data is stored in the Parquet format for efficient compression and faster loading times.

## License

This project is open source, available under the MIT License.
