
import numpy as np
import pandas as pd
import os

import yaml

from sklearn.model_selection import train_test_split

# loading our test_size value from yaml file
test_size = yaml.safe_load(open('params.yaml', 'r'))['data_ingestion']['test_size']

#code to bring data from url or data source
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# we will do certain changes - preporcessing
df.drop(columns=['tweet_id'], inplace=True)

final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# also we will do train test split
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

# we will maintain a raw copy of data inside data/raw/train.csv and data/raw/test.csv
data_path = os.path.join("data", "raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"))
test_data.to_csv(os.path.join(data_path, "test.csv"))