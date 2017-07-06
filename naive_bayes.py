# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# print(X)
# print(y)
X = np.array([[]])
X = None
y = np.array([])
y = []
with open('Trainingsdaten.pat') as f:
    dim_x = int(f.readline().strip())
    dim_y = int(f.readline().strip())
    foo = int(f.readline().strip())
    target_dimensions = int(f.readline().strip())

    for i in range(0, 260):
        data = []
        for j in range(0, 14):
            data.append(f.readline().strip())
        # data = " ".join(data).split()
        data = list(map(float, " ".join(data).split()))

        if X is None:
            X = np.array([data])
        else:
            X = np.append(X, [data], axis=0)

        tmp = f.readline().strip().split()
        # print(tmp)
        print(tmp.index("0.80"))
        y.append(tmp.index("0.80"))
        print(y)

    y = np.array(y)
    print(y)
    # exit()
    # print(X)
    # print(y)
    # exit()
# dataset = pd.read_csv('Trainingsdaten.csv')
# X = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:, 4].values

# print(len(X))
# print(X)
# print(len(y))
# print(y)
# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# X_train = X
# y_train = y
# X_test = X
# y_test = y
# print(len(X_train))
# print(len(y_train))
# print(X_train)
# print(y_train)

# print(len(X_test))
# print(len(y_test))

# exit()



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()