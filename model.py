from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

import joblib

#DATA
heart_data = pd.read_csv('heart.csv')

#Check to see missing data ----> No missing data
#heart_data.isnull().sum()

#Assigned data
X = heart_data.drop('target', axis = 1)
y = heart_data['target']


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .18, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(random_state = 0)

logistic_model.fit(X_train_scaled, y_train)


predictions = logistic_model.predict(X_test_scaled)

joblib.dump(logistic_model, 'logistic_model.pkl')