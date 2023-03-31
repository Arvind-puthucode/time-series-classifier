import pandas as pd 
from sarimax import sarimax_model
from linear_regression import lrModel
from exps import exps_model
class TimeSeriesModel:
    def __init__(self, data: pd.Series):
        self.data = data
    def create_all_models(self):
        self.errors={}
        sarimax_instance=sarimax_model(self.data.copy())
        mape_s=sarimax_instance.create_model()
        self.errors['sarimax']=mape_s
        lr_instance=lrModel(self.data)
        mape_lr=lr_instance.create_model()
        self.errors['lr']=mape_lr
        exps_instance=exps_model(self.data)
        mape_exps=exps_instance.create_model()
        self.errors['exps']=mape_exps
        print(f'the errros:',self.errors)
        # best model get 
        return min(self.errors)
    

if __name__ == "__main__":
    data = pd.read_csv("sample_1.csv")
    data.drop(['ind'],axis=1,inplace=True)
    data=data.set_index('point_timestamp')
    t1=TimeSeriesModel(data)
    print(t1.create_all_models())
