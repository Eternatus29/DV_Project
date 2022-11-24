import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

# Load the csv file
df = pd.read_csv("FuelConsumption.csv")
df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

print(df.head())

y_data = df.iloc[:, 3].values
x_data = df.iloc[:, [0,1,2]].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)
yhat_test = clf.predict(x_test)
yhat_train = clf.predict(x_train)


# Make pickle file of our model
pickle.dump(clf, open("model.pkl", "wb"))

import os
os.getcwd()