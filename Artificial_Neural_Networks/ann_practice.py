# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#TODO seems column order changed  (add to first column)
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# convert/encode country and gender into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
# normalize encoded country number
# TODO seems columnorder changed again!, (add to first column)
X = onehotencoder.fit_transform(X).toarray()
# remove dummy variables
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
# test_size = 0.2: 20% data will be used for testing, 80% data will be used for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# TODO why test data no need to scaling
X_test = sc.transform(X_test)

# part 2 build ann
import keras
from keras.models import Sequential
from keras.layers import Dense
