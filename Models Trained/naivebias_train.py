import pickle
import random
import joblib as jl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
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

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

jl.dump(scaler, '../Extra/std_scaler.bin', compress=True)


# transform input data into a non-negative format
vectorizer = CountVectorizer()
X_train = [' '.join(map(str, x)) for x in X_train]
X_test = [' '.join(map(str, x)) for x in X_test]
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)


# Define a parameter grid for the Multinomial Naive Bayes model
param_grid = {'alpha': [0.1, 1, 10, 100]}

# Define the Multinomial Naive Bayes model
nb = MultinomialNB()

# Chooses the best parameters from param_grid for the Multinomial Naive Bayes model
grid_search = GridSearchCV(nb, param_grid, cv=5, verbose=10)

start_time = time.time()

# Trains the model on the specified training data
grid_search.fit(X_train_counts, Y_train)

end_time = time.time()
print(f"Training time: {end_time-start_time:.2f} seconds")

# Prints the best parameters that the model chose for the given data
nb_tuned = grid_search.best_estimator_
print(nb_tuned)

# Saves the model in 'model.sav' folder
pick = open('../Saved Models/nb_model.sav', 'wb')
pickle.dump(nb_tuned, pick)
pick.close()
pick = open('../Saved Models/nb_model.sav', 'rb')
model = pickle.load(pick)
pick.close()

# perform predictions of X_test
model_predictions = model.predict(X_test_counts)

# Print out a classification report
print(classification_report(Y_test, model_predictions))
