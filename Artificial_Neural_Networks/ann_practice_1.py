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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling, normalized to 0-1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# NOTE: apply same transform to test
X_test = sc.transform(X_test)

# part 2 build ann
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
""" NOTE: number of neural units = 6 is suggested by (input_dim=11 + output_dim=1)/2 = (11+1)/2 = 6"""
# TODO after some experiment, it seems accuracy increased if units increased to 10 - 16
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
""" NOTE: sigmoid good for 0-1 possibility output """
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
""" NOTE: epochs is number of iterations """
classifier.fit(X_train, y_train, batch_size = 10, epochs = 40)


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0][0] + cm[1][1])/y_test.size;
print ('Test Accuracy = {}'.format(accuracy) )


# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

exampleX = np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
new_prediction = classifier.predict(sc.transform(exampleX))
print ('Example X will leave = {} as {}'.format(new_prediction, new_prediction > 0.5))