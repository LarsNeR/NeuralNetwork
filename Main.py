from NeuralNetwork import NeuralNetwork
from sklearn import datasets, preprocessing, model_selection
from sklearn.metrics import precision_recall_fscore_support

# Load dataset (you can also load your own with pandas but sklearn offers
# a range of different datasets)
(X, y) = datasets.load_breast_cancer(return_X_y=True)

# Preprocess it with sklearn (not necessary, but improves finding minimum)
X = preprocessing.scale(X)

# Divide dataset into train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65)

# Instantiate a new Neural Network and call its fit-method with the
# train data
nn = NeuralNetwork(n_hidden_nodes=5)
nn.fit(X=X_train, y=y_train)

# Predict the result with the test data and calculate precision, recall
# and fscore
y_pred = nn.predict(X_test)
(precision, recall, fscore, _) = precision_recall_fscore_support(y_test, y_pred)

# Print interesting information
print(precision, recall, fscore)
