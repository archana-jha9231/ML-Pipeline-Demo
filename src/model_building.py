import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier

import yaml
load_yaml = yaml.safe_load(open('params.yaml', 'r'))
n_estimators = load_yaml['model_building']['n_estimators']
learning_rate = load_yaml['model_building']['learning_rate']


# fetch the data from data/features
train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the XGBoost model

clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open('model.pkl','wb'))