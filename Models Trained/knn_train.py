import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib as jl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
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
output = open('../Extra/encoder.pkl', 'wb')
pickle.dump(label_encoder, output)
output.close()

# Separate the data into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3)


#scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
jl.dump(scaler, '../Extra/std_scaler.bin', compress=True)

#Grid search best k-NN parameters
parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}


# create an instance of the knn classifier
knn_grid_tuned = KNeighborsClassifier()

# create an instance of grid search with the above parameters
grid_search = GridSearchCV(knn_grid_tuned, parameters, cv=5, scoring='accuracy', return_train_score=True, verbose=10)

start_time = time.time()

# fit the grid search with training set
grid_search.fit(X_train_scaled, Y_train)

end_time = time.time()

print(f"Training time: {end_time-start_time:.2f} seconds")

# retrieve the best estimator
knn_tuned = grid_search.best_estimator_
print(knn_tuned)

print(accuracy_score(Y_test, knn_tuned.predict(X_test_scaled)))

# Saves the model in 'knn_model.sav' folder
pick = open('../Saved Models/knn_model.sav', 'wb')
pickle.dump(knn_tuned, pick)
pick.close()

