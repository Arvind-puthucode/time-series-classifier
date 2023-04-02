import pandas as pd 
from sarimax import sarimax_model
from linear_regression import lrModel
from exps import exps_model
import concurrent.futures


class TimeSeriesModel:
    def __init__(self, data: pd.Series):
        self.data = data

    def create_all_models(self):
        """returns name of best model with mape error"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the tasks to the executor
            future_to_model = {
                executor.submit(sarimax_model(self.data.copy()).create_model): 'sarimax',
                executor.submit(lrModel(self.data).create_model): 'lr',
                executor.submit(exps_model(self.data).create_model): 'exps'
            }
            # Wait for the tasks to complete and retrieve the results
            errors = {}
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    mape = future.result()
                    if isinstance(mape, pd.Series):
                        mape = float(mape)
                    errors[model_name] = mape
                except Exception as e:
                    print(f'{model_name} model failed with error: {e}')
        print(f'The errors: {errors}')
        # Get the name of the model with the lowest error
        best_model = min(errors, key=lambda k: errors[k])
        return best_model, errors[best_model]

if __name__ == "__main__":
    data = pd.read_csv("sample_1.csv")
    data.drop(['ind'],axis=1,inplace=True)
    data=data.set_index('point_timestamp')
    t1=TimeSeriesModel(data)
    print(t1.create_all_models())
