# Data Directory

This directory contains data files used by the Supply Chain Scenario Planner.

## Directory Structure

- `raw/` - Raw data files before processing
- `processed/` - Cleaned and processed data files
- `scenarios/` - Saved scenario configurations and results
- `models/` - Trained model files and checkpoints

## Data Files

The following data files are ignored by git to prevent large files from being tracked:
- CSV files (*.csv)
- Excel files (*.xlsx, *.xls)
- JSON files (*.json)
- Parquet files (*.parquet)
- Feather files (*.feather)
- Pickle files (*.pickle, *.pkl)

## Adding Data

To add new data files:
1. Place raw data files in the `raw/` directory
2. Use the data processing scripts to clean and transform the data
3. Processed files will be saved in the `processed/` directory

## Note

This README.md file is tracked by git, but the data files themselves are not tracked to keep the repository size manageable.
