# Created by William Gunn u3258398
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

# Data file import
asteroid_data = pd.read_csv("neo2_v2.csv")

# Attribute to be predicted
predict = "hazardous"

# pre-processing
# encode object columns to integers
for col in asteroid_data:
    if asteroid_data[col].dtype == 'object':
        asteroid_data[col] = OrdinalEncoder().fit_transform(asteroid_data[col].values.reshape(-1, 1))

# Dataset to be Predicted, X is all attributes and y is the features
le = preprocessing.LabelEncoder()
id = le.fit_transform(list(asteroid_data["id"]))
name = le.fit_transform(list(asteroid_data["name"]))
est_diameter_min = le.fit_transform(list(asteroid_data["est_diameter_min"]))
est_diameter_max = le.fit_transform(list(asteroid_data["est_diameter_max"]))
relative_velocity = le.fit_transform(list(asteroid_data["relative_velocity"]))
miss_distance = le.fit_transform(list(asteroid_data["miss_distance"]))
orbiting_body = le.fit_transform(list(asteroid_data["orbiting_body"]))  # true = 0 false = 1
sentry_object = le.fit_transform(list(asteroid_data["sentry_object"]))  # true = 0 false = 1
absolute_magnitude = le.fit_transform(list(asteroid_data["absolute_magnitude"]))
hazardous = le.fit_transform(list(asteroid_data["hazardous"]))  # not hazardous = 0 hazardous = 1

x = list(zip(est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude))
y = list(hazardous)

# Model Test/Train
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = "accuracy"
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
# 0.2 means 80% training 20% testing, with higher data it already has seen that information and knows
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)


# Prediction class for import
@dataclass(eq=True, frozen=True, order=True)
class Prediction:
    best_model = RandomForestClassifier()
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_pred)
