"""
    classifier algo     
"""
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from parameters_test import Parameters
from time_series_models import TimeSeriesModel
from sklearn.tree import plot_tree

# Path to the data folder
def self_training_params():
    main_dir="/mnt/c/time-series-classifier/Sample-Time-Series"
    params_data=[]
    subfolders = ['daily', 'hourly', 'weekly', 'monthly']
    dfs = {}
    for subfolder in subfolders:
        # Get the path to the subfolder
        subfolder_path = os.path.join(main_dir, subfolder)
        print('\n sub_path \n',subfolder_path,'\n')
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            df_name = f'{subfolder}\{csv_file[:-4]}'
            print(f'\n\n df_name is {df_name}\n\n',)
            df = pd.read_csv(os.path.join(subfolder_path, csv_file),index_col=0)
            df=df.set_index(df.columns[0])
            df= df.fillna(df.mean())

            print(df_name,df.head(1)) 
            p1=Parameters(df)
            timeseries = (df)[df.columns[0]]
            params=p1.get_params(timeseries)
            best_model,error_model=TimeSeriesModel(df).create_all_models()
            params['best_model']=best_model
            params_data.append(params)
def training_another():
    root_dir = "/mnt/c/time-series-classifier"
    main_dir="/mnt/c/time-series-classifier/Sample-Time-Series"
    params_data=[]
    subfolders = ['lr_series', 'exps_series', 'sarimax_series']
    
    dfs = {}
    for subfolder in subfolders:
        # Get the path to the subfolder
        subfolder_path = os.path.join(main_dir, subfolder)
        print('\n sub_path \n',subfolder_path,'\n')
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            df_name = f'{subfolder}\{csv_file[:-4]}'
         #   print(f'\n\n df_name is {df_name}\n\n',)
            df = pd.read_csv(os.path.join(subfolder_path, csv_file),index_col=0)
            df=df.set_index(df.columns[0])
            df= df.fillna(df.mean())

            #print(df_name,df.head(1)) 
            p1=Parameters(df)
            timeseries = (df)[df.columns[0]]
            print(csv_file)
            params=p1.get_params(timeseries)
            params['best_model']=subfolder
            params_data.append(params)
    df = pd.DataFrame(params_data)
    df.to_csv(f'{main_dir}/train_params.csv')
# Define the classifier
def testing_data():
    
    train_data=pd.read_csv('train_params.csv')
    train_data = train_data.sample(frac=1)
    
    #print(train_data)
    X = train_data.iloc[:, 1:-1]
   # print('X',X)
    y = train_data.iloc[:, -1]
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    classifier.fit(X,y)
    # Print the decision tree for the first tree in the forest
    # Visualize the decision tree for the first tree in the forest
    plt.figure(figsize=(20,10))
    plot_tree(classifier.estimators_[0], 
            feature_names=X.columns, 
            class_names=y.name, 
            filled=True)
    plt.show()
    # Convert the dot file to a png image using the Graphviz package
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    # Display the decision tree image
    from IPython.display import Image
    Image(filename = 'tree.png')
    
    test_data=pd.read_csv("test_params.csv")
    X_test = test_data.iloc[:, 1:-1]
    print('x test',X_test)
    y_test = test_data.iloc[:, -1]
    y_pred=classifier.predict(X_test)
    print(y_pred,"ypred")
    
if __name__=="__main__":
    main_dir="/mnt/c/time-series-classifier"
    testing_data()
    #params_arr=training_another()
    