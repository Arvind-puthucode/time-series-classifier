import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV,train_test_split

class lrModel:
    def __init__(self,data:pd.DataFrame):
        self.data=data
        X = np.arange(len(self.data)).reshape(-1, 1)
       # print('x',X)
        y = self.data[data.columns[0]]
        print('y',y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def hyper_parameter_optimization(self):
        # Define the parameter grid
        param_grid = {'fit_intercept': [True, False],'n_jobs':[1,2,3,4] , 'copy_X': [True, False],'positive':[True,False]}
        # Create the grid search object
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
        # Fit the grid search to the training data
        grid_search.fit(self.X_train, self.y_train)
        # Get the best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Train the model with the best parameters
        print('best score of the optmized lr', best_score)
        lr = LinearRegression(**best_params)
        return lr

    def create_model(self):
        model=self.hyper_parameter_optimization()
        # test the model
        model.fit(self.X_train, self.y_train)
        y_pred=model.predict(self.X_test)
        return self.mape(y_pred)
    
         
    def mape(self,y_pred):
        abs_diffs = np.abs(self.y_test - y_pred)
        pct_diffs = abs_diffs / self.y_test
        # handle divide-by-zero errors (replace NaNs with 0)
        pct_diffs[np.isnan(pct_diffs)] = 0
        # calculate mean of percentage differences
        MAPE_error = np.mean(pct_diffs) * 100
        return MAPE_error
if __name__ == "__main__":
    eg_data = pd.read_csv("sample_1.csv")
    linear_model=lrModel(eg_data)
    print(f'mape error is{linear_model.create_model()}')
