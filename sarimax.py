import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from skforecast.model_selection_statsmodels import grid_search_sarimax
import itertools
import statsmodels.api as sm
class sarimax_model:
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
        pdq,pdqs=self.hyper_parameter_optimization()
        print('pdq,pdqs',pdq,pdqs)
        model=sm.tsa.statespace.SARIMAX(endog=self.y_train,order=pdq,seasonal_order=pdqs)
        out=model.fit()
        y_pred=out.predict(start=self.X_test[0],end=self.X_test[-1])
       # print(y_pred,"y_pred")
        return self.mape(y_pred)   
         
    def mape(self,y_pred):
        abs_diffs = np.abs(self.y_test - y_pred)
        pct_diffs = abs_diffs / self.y_test
        # handle divide-by-zero errors (replace NaNs with 0)
        pct_diffs[np.isnan(pct_diffs)] = 0
        # calculate mean of percentage differences
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error
    
    def hyper_parameter_optimization(self):
        ### Define Parameter Ranges to Test ###
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        ### Run Grid Search ###
        # Define function
        print('pdsq',pdqs)
        def sarimax_gridsearch(ts, pdq, pdqs,maxiter=200, freq='D'):
            '''
            Input: 
                ts : your time series data
                pdq : ARIMA combinations from above
                pdqs : seasonal ARIMA combinations from above
                maxiter : number of iterations, increase if your model isn't converging
                frequency : default='D' for day. Change to suit your time series frequency
                    e.g. 'D' for day, 'H' for hour, 'Y' for year. 
                
            Return:
                Prints out top 5 parameter combinations
                Returns dataframe of parameter combinations ranked by BIC
            '''

            # Run a grid search with pdq and seasonal pdq parameters and get the best BIC value
            ans = []
            count=0
            for comb,combs in zip(pdq,pdqs):
                try:
                    count+=1
                    if count>=6:
                        break
                    mod = sm.tsa.statespace.SARIMAX(endog=ts, # this is your time series you will input
                                                    order=comb,
                                                    seasonal_order=combs,
                                                    #enforce_stationarity=False,
                                                    #enforce_invertibility=False,
                                                    #freq=freq
                                                    )
                    output = mod.fit() 
                    y_pred=output.predict(start=self.X_test[0], end=self.X_test[-1])
                    error=self.mape(y_pred)
                    ans.append([comb, combs,output.bic])
                    print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(comb, combs, output.bic))
                except:
                    print('exception error')
                    continue
            # Find the parameters with minimal BIC value
            # Convert into dataframe
            ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'bic'])
            # Sort and return top 5 combinations
            ans_df = ans_df.sort_values(by=['bic'],ascending=True)[0:5]
            return ans_df
        ### Apply function to your time series data ###
        # Remember to change frequency to match your time series data
        ans1=sarimax_gridsearch(self.data, pdq, pdqs,freq='D')
        best_pdq=ans1['pdq'][0]
        best_pdqs=ans1['pdqs'][0]
        print('best pdq:', best_pdq,type(best_pdq),'best pdqs:', best_pdqs,type(best_pdqs))
        return best_pdq, best_pdqs


if __name__ == "__main__":
    eg_data = pd.read_csv("sample_1.csv")
    eg_data.drop(['ind'],axis=1,inplace=True)
    #print('eg_data',eg_data)
    eg_data=eg_data.set_index('point_timestamp')
    #print('eg_data',eg_data)
    sarimax=sarimax_model(eg_data)
    print(f'arima mape error is{sarimax.create_model()}')
   
        