import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
# Set the number of datasets to generate
n_datasets = 100

# Set the number of data points per dataset
n_points = 400

# Set the range of x values
x_min = 0
x_max = 10000

# Set the noise levels to test
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
main_dir = "/mnt/c/time-series-classifier/Sample-Time-Series"

timestamp_min = pd.Timestamp('2019-01-01')
timestamp_max = pd.Timestamp('2022-01-01')

# Set the orders for the SARIMAX model
orders = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)]

# Loop over the noise levels and generate datasets
for noise_level in noise_levels:
    for i in range(n_datasets):
        # Generate x values
        x = np.linspace(x_min, x_max, n_points)
        # Generate timestamp values
        timestamps = pd.date_range(start=timestamp_min, end=timestamp_max, periods=n_points)
        
        # Generate y values with seasonal and trend components and added noise
        y = 2*np.sin(2*np.pi*x/4) + 4*np.sin(2*np.pi*x/10) + x + np.random.normal(scale=noise_level, size=n_points)
        
        # Set the orders for the SARIMAX model
        # Set the orders for the SARIMAX model
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        # Set the seasonal orders for the SARIMAX model
        seasonal_orders = [(1, 0, 0, 4), (0, 1, 0, 4), (0, 0, 1, 4), (1, 1, 0, 4), (0, 1, 1, 4), (1, 0, 1, 4), (1, 1, 1, 4)]
        random_pair=random.sample(list(zip(orders, seasonal_orders)),1)[0]
        print('random_pairs',random_pair)
        order,seasonal_order=random_pair[0],random_pair[1]
        model = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit()
        
        # Create DataFrame with x, y, and noise level columns
        df = pd.DataFrame(
            {'ind': range(n_points), 'timestamp': timestamps, 'point_value': y})
        
        # Save DataFrame to CSV file
        filename = f"{main_dir}/sarimax_series/dataset_sarimax_{noise_level}_{i}.csv"
        df.to_csv(filename, index=False)
