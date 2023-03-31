import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pywt
from scipy.stats import entropy

from statsmodels.tsa.stattools import adfuller, acf


class Paramaters:
    def __init__(self, data_set: pd.DataFrame):
        self.data = data_set
    # statistical measures

    def get_cv(self, timeseries):
        """Function to calculate the coefficient of variation (CV) of a given time series"""
        mean = np.mean(timeseries)
        std = np.std(timeseries)
        cv = std / mean
        return cv

    # Define a function to test for stationarity using the ADF test
    def test_stationarity(self, timeseries):
        # Perform ADF test
        result = adfuller(timeseries)
        adf_stat = result[0]
        # use critical value for 1% significance level
        crit_val = result[4]['1%']
        norm_adf_stat = np.abs(adf_stat / crit_val)
        return norm_adf_stat

    # measure for trend ie slope as a indicator

    def measure_trend(self, timeseries):
        # Fit linear regression to the time series
        X = np.arange(len(timeseries)).reshape(-1, 1)
        y = np.array(timeseries)
        reg = LinearRegression().fit(X, y)

        # Calculate the slope of the regression line
        slope = reg.coef_[0]

        # Normalize the slope value between -1 and 1 using min-max scaling
        min_slope = np.min(reg.coef_)
        max_slope = np.max(reg.coef_)
        norm_slope = (slope - min_slope) / (max_slope - min_slope)

        return norm_slope

    """auto correlation features
    """

    def auto_corr_measure(self, timeseries):
        # calculate the ACF
        acf_values, confint = acf(timeseries, nlags=10, alpha=0.05, fft=False)
        max_autocorr = np.max(np.abs(acf_values))
        return max_autocorr

    def seasonality_measure(self, timeseries):
        # calculate the power spectrum
        fft_values = np.fft.fft(timeseries)
        power_spectrum = np.abs(fft_values) ** 2

        # identify the dominant frequency
        max_freq_index = np.argmax(power_spectrum[1:len(timeseries)//2]) + 1
        dominant_freq = max_freq_index / len(timeseries)

        # calculate the ratio of power at the dominant frequency to total power
        power_ratio = np.max(power_spectrum) / np.sum(power_spectrum)
        return power_ratio

    def wavelet_rms(self, timeseries):
        # perform wavelet transform
        coeffs = pywt.wavedec(timeseries, 'db1')
        # calculate RMS of coefficients
        rms = 0
        for i in range(len(coeffs)):
            rms += (sum(j*j for j in coeffs[i]))/len(coeffs[i])
        rms = rms**(1/2)
        # Calculate maximum possible RMS value
        max_val = np.sqrt(len(timeseries) * np.max(timeseries)**2)
        # Normalize the RMS value
        norm_rms = rms / max_val
        return norm_rms
    
    def measure_entropy(self, timeseries):
        # Calculate the histogram of the time series
        hist, _ = np.histogram(timeseries, bins='auto', density=True)

        # Calculate the Shannon entropy of the histogram
        entropy_val = entropy(hist)

        return entropy_val


if __name__ == "__main__":
    eg_data = pd.read_csv("data.csv")
    eg1 = Paramaters(eg_data)
    timeseries = (eg1.data)[eg_data.columns[1]]
    parameters={'p1': eg1.get_cv(timeseries),
                       "p2": eg1.test_stationarity(timeseries),
                       "p3": eg1.measure_trend(timeseries),
                       "p4": eg1.auto_corr_measure(timeseries),
                       "p5": eg1.seasonality_measure(timeseries),
                       "p6": eg1.wavelet_rms(timeseries),
                       "p7": eg1.measure_entropy(timeseries),
                }
    print(parameters, "parameters")
