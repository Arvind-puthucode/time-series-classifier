import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Set the number of datasets to generate
n_datasets = 100

# Set the number of data points per dataset
n_points = 500
main_dir="/mnt/c/time-series-classifier/Sample-Time-Series"

# Set the range of timestamp values
timestamp_min = pd.Timestamp('2019-01-01')
timestamp_max = pd.Timestamp('2022-01-01')

# Set the noise levels to test
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

# Loop over the noise levels and generate datasets
for noise_level in noise_levels:
    for i in range(n_datasets):
        # Generate timestamp values
        timestamps = pd.date_range(start=timestamp_min, end=timestamp_max, periods=n_points)
        
        # Generate point values with linear trend and added noise
        x = np.linspace(0, 10, n_points)
        y = 2*x + 1 + np.random.normal(scale=noise_level, size=n_points)
        
        # Fit linear regression model
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        
        # Compute R-squared value
        r2 = lr.score(x.reshape(-1, 1), y)
        
        # Create DataFrame with timestamp, point_value, and noise level columns
        df = pd.DataFrame({'ind': range(n_points),'timestamp': timestamps, 'point_value': y})
        
        # Save DataFrame to CSV file
        filename = f"{main_dir}/lr_series/dataset_lr_{noise_level}_{i}.csv"
        df.to_csv(filename, index=False)
