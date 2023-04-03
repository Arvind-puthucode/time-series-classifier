import pandas as pd 
from parameters_test import Parameters
key=3
df=pd.read_csv('test_params.csv')
tst_df=pd.read_csv('Sample-Time-Series/test_data/test_3.csv')
tst_df.drop(['ind'],axis=1,inplace=True)
tst_df=tst_df.set_index('timestamp')
t1=(tst_df)[tst_df.columns[0]]
  
params=Parameters(tst_df).get_params(t1)
df=pd.concat([df,pd.DataFrame(params,index=[key])])
df.to_csv('test_params.csv')
