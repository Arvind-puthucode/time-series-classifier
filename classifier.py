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
def training_params():
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
# Define the classifier
if __name__=="__main__":
    train_data=pd.read_csv('trained_data.csv')
    #print(train_data)
    X = train_data.iloc[:, :-1]
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