import pickle
import random
import joblib as jl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time

pick_in = open('../Preprocessing/data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature.flatten())
    labels.append(label)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Separate the data into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3)

#scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
jl.dump(scaler, '../Extra/std_scaler.bin', compress=True)


# Define a parameter grid for the SVM model
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.01, 0.001, 0.0001],
              'kernel': ['poly', 'linear']}

# Define the SVM model
svc = svm.SVC(probability=True)

# Chooses the best parameters from param_grid for the SVM model
grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=10)


start_time = time.time()

# Trains the model on the specified training data
grid_search.fit(X_train, Y_train)

end_time = time.time()

print(f"Training time: {end_time-start_time:.2f} seconds")

# Prints the best parameters that the model chose for the given data
svm_tuned = grid_search.best_estimator_
print(svm_tuned)

# Saves the model in 'model.sav' folder
pick = open('svm_model.sav', 'wb')
pickle.dump(svm_tuned, pick)
pick.close()
pick = open('svm_model.sav', 'rb')
model = pickle.load(pick)
pick.close()

# Testing phase: predict and store the predictions of the testing data in model_predictions
model_predictions = model.predict(X_test)


# Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
print(classification_report(Y_test, model_predictions))