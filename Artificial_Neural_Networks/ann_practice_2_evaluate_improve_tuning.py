# Artificial Neural Network

# Part 1 - Data Preprocessing
import pandas as pd
import numpy as np

# Importing the dataset
"""NOTE: column order changed  (add to first column)"""
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# convert/encode country and gender into numbers (0 or 1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])

"""NOTE: column order changed here, (add to first column)"""
# normalize encoded country number
X = onehotencoder.fit_transform(X).toarray()
# remove dummy variables
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
# test_size = 0.2: 20% data will be used for testing, 80% data will be used for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Feature Scaling, normalized to 0-1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# NOTE: apply same transform to test
X_test = sc.transform(X_test)

# Part 4 - Evaluating, Improving and Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    """ apply dropout to avoid over fitting """
    # classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    """ apply dropout to avoid over fitting """
    # classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 40)
""" cv = 10 means run 10 times/folds """
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

print ('Test accuracies: mean = {}, variance = {}'.format(mean,variance ))

""" 
KerasClassifier is only a method to try and estimate the error. 
For predicting, you need to make the classifier and train it again on the data and then use predict.
"""