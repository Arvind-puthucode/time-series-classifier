import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


class TimeSeriesModel:
    def __init__(self, data: pd.Series):
        self.data = data
        train_size = int(len(self.data) * 0.8)
        self.training_data,self.test_data = self.data[0:train_size].values.reshape(-1, 1), self.data[train_size:len(self.data)].values.reshape(-1,1)
    def mape(self,y_pred):
        abs_diffs = np.abs(self.test_data - y_pred)
        pct_diffs = abs_diffs / self.test_data
        # handle divide-by-zero errors (replace NaNs with 0)
        pct_diffs[np.isnan(pct_diffs)] = 0
        # calculate mean of percentage differences
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error

    def arima(self, order):
        model =ARIMA(self.training_data, order=(1,1,2))
        result = model.fit()
        train_size = int(len(self.data) * 0.8)

        y_pred=result.predict(start=train_size,end=len(self.data))
        return self.mape(y_pred)


    def linear_regression(self):
        X = np.arange(len(self.training_data)).reshape(-1, 1)
       # print('x',X)
        y = self.training_data
        #print('y',y)
        model = LinearRegression()
        model.fit(X,y)
        # test the model
        x_test=np.arange(len(self.test_data)).reshape(-1, 1)
        y_pred=model.predict(x_test)
        return self.mape(y_pred)
        
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    #print(data)
    ts_model = TimeSeriesModel(data['Value'])
    arima_mape = ts_model.arima((3, 1, 0))
    lr_mape = ts_model.linear_regression()
    print("ARIMA MAPE:", arima_mape)
    print("Linear Regression MAPE:", lr_mape)
