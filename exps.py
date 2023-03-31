import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import statsmodels.tsa as smts
import warnings
class exps_model:
    def __init__(self,df:pd.DataFrame):
        self.data=df
        l=len(self.data)
        X = self.data.index
        #print('x',X)
        y = self.data[df.columns[0]]
        #print('y',y)
        limit=int(l*0.7)
        self.X_train, self.X_test, self.y_train, self.y_test = X[0:limit],X[limit:],y[0:limit],y[limit:]
        print(f'xtrain{self.X_train[0:5]}ytrain:{self.y_train}xtest:{self.X_test}ytest{self.y_test}')
    def create_model(self):
        params=self.hyper_parameter_optimization()
        print('best params',params)
        trend = params.get('trend')
        seasonal=params.get('seasonal')
        seasonal_periods = params.get('seasonal_periods')
        initialization_method=params.get("initialization_method")
        damped_trend = params.get('damped_trend')
        use_boxcox = params.get('use_boxcox')
        return params.get('mape')    
    def mape(self,y_pred):
        abs_diffs = np.abs(self.y_test.values.astype(np.float) - y_pred.values.astype(np.float))
        pct_diffs = abs_diffs / self.y_test
        # handle divide-by-zero errors (replace NaNs with 0)
        pct_diffs[np.isnan(pct_diffs)] = 0
        # calculate mean of percentage differences
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error
    
    def hyper_parameter_optimization(self):
        ### Define Parameter Ranges to Test ###
        from sklearn.model_selection import ParameterGrid
        param_grid = {'trend': [None,'add', 'mul'],'seasonal' :['add', 'mul'],'seasonal_periods':[3,6,12]
                      ,'initialization_method':[None,'estimated','heuristic','legacy-heuristic','known']
                      ,'damped_trend' : [True, False], 
                      'use_boxcox':[True, False],
                    }
        pg = list(ParameterGrid(param_grid))
        df_results_moni = pd.DataFrame(columns=['trend','seasonal', 'seasonal_periods', 
                                            'initialization_method', 'damped_trend', 'use_boxcox','mape'])
        start = time.time()
        print('Starting Grid Search..')
        print(f'len of # of parameters to be searched',len(pg))
        for a,b in enumerate(pg):
            try:
                trend = b.get('trend')
                seasonal=b.get('seasonal')
                seasonal_periods = b.get('seasonal_periods')
                initialization_method=b.get("initialization_method")
                damped_trend = b.get('damped_trend')
                use_boxcox = b.get('use_boxcox')
                if (damped_trend and not trend):
                    continue

                #print(trend,smoothing_level, smoothing_slope,dampend_trend,use_boxcox,remove_bias,use_basinhopping)
                fit1 = smts.holtwinters.ExponentialSmoothing(endog=self.y_train,trend=trend,initialization_method=initialization_method,damped_trend=damped_trend,
                                            seasonal=seasonal,seasonal_periods=seasonal_periods,use_boxcox=use_boxcox).fit()
                #fit1.summary
                y_pred = fit1.predict(start=self.X_test[0],end=self.X_test[-1])
                df_pred = pd.DataFrame(y_pred, columns=['Forecasted_result'])
                #print('predicted vs test',y_pred,self.y_test)
                mape_error=self.mape(y_pred)
                #print( f' RMSE is {np.sqrt(metrics.mean_squared_error(test, df_pred.Forecasted_result))}')
         #       print('mape_error is ',mape_error)
                df_results_moni = df_results_moni.append({'trend':trend,'seasonal':seasonal,'seasonal_periods':seasonal_periods,
                                                    'initialization_trend':initialization_method,
                                        'damped_trend':damped_trend,'use_boxcox':use_boxcox,'mape':mape_error}, ignore_index=True)  
                #print('appended',a,df_results_moni)
            except ValueError:
                print('skipped this parameter',a,b)
                continue
        print('End of Grid Search')
        end = time.time()
        print(f' Total time taken to complete grid search in seconds: {(end - start)}')
        best_res_params=df_results_moni.sort_values(by=['mape']).head(1)
       # print('best_res_params',best_res_params)
        return best_res_params
    



if __name__ == "__main__":
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eg_data = pd.read_csv("sample_1.csv")
        eg_data.drop(['ind'],axis=1,inplace=True)
        #print('eg_data',eg_data)
        eg_data=eg_data.set_index('point_timestamp')
        #print('eg_data',eg_data)
        exps=exps_model(eg_data)
        print(f'arima mape error is{exps.create_model()}')
    
            