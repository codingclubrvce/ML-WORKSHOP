# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
# metric can be euclidean/manhattan/minkowski
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
classifier.fit(X_train, y_train)