import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set the number of datasets to generate
n_datasets = 100

# Set the number of data points per dataset
n_points = 1000

# Set the range of x values
x_min = 0
x_max = 100

# Set the noise levels to test
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
main_dir="/mnt/c/time-series-classifier/Sample-Time-Series"

# Set the range of timestamp values
timestamp_min = pd.Timestamp('2019-01-01')
timestamp_max = pd.Timestamp('2022-01-01')

# Loop over the noise levels and generate datasets
for noise_level in noise_levels:
    for i in range(n_datasets):
        # Generate x values
        x = np.linspace(x_min, x_max, n_points)
        # Generate timestamp values
        timestamps = pd.date_range(start=timestamp_min, end=timestamp_max, periods=n_points)
        
        # Generate y values with exponential trend and added noise
        y = np.exp(0.2 * x) + np.random.normal(scale=noise_level, size=n_points)
        
        # Fit exponential smoothing model
        model = ExponentialSmoothing(y)
        fitted_model = model.fit()
        
        # Compute R-squared value
        r2 = fitted_model.sse
        
        # Create DataFrame with x, y, and noise level columns
        df = pd.DataFrame({'ind': range(n_points), 'timestamp': timestamps, 'point_value': y})
        
        # Save DataFrame to CSV file
        filename = f"{main_dir}/exps_series/dataset_exp_{noise_level}_{i}.csv"
        df.to_csv(filename, index=False)
