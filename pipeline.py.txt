# Import the required libraries

from kfp.dsl import component, pipeline
import kfp
from kfp import kubernetes

# Data Preparation

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def prepare_data(data_path: str):
    import pandas as pd
    import os
    from sklearn import datasets
    
    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    df = df.dropna()
    df.to_csv(f'{data_path}/final_df.csv', index=False)

# Split data into Train and Test set

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def train_test_split(data_path: str):    
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    
    final_data = pd.read_csv(f'{data_path}/final_df.csv')
    
    target_column = 'species'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=47)
    
    np.save(f'{data_path}/X_train.npy', X_train)
    np.save(f'{data_path}/X_test.npy', X_test)
    np.save(f'{data_path}/y_train.npy', y_train)
    np.save(f'{data_path}/y_test.npy', y_test)

# Model training

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def training_basic_classifier(data_path: str):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.linear_model import LogisticRegression
    
    X_train = np.load(f'{data_path}/X_train.npy',allow_pickle=True)
    y_train = np.load(f'{data_path}/y_train.npy',allow_pickle=True)
    
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train,y_train)
    import pickle
    with open(f'{data_path}/model.pkl', 'wb') as f:
        pickle.dump(classifier, f)