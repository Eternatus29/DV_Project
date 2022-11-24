import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

# Load the csv file
df = pd.read_csv("FuelConsumption.csv")

print(df.head())

# Select independent and dependent variable
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = df[['CO2EMISSIONS']]
regr.fit(x,y)


# Make pickle file of our model
pickle.dump(regr, open("model.pkl", "wb"))

import os
os.getcwd()