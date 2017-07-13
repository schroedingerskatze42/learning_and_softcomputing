# Naive Bayes

# Importing the libraries
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


def read_data(file_name):
    X = None
    y = None
    with open(file_name) as f:
        # meta information
        dim_x = int(f.readline().strip())
        dim_y = int(f.readline().strip())
        __foo = int(f.readline().strip())
        target_dimensions = int(f.readline().strip())

        for l in f:
            # read data
            data = [l]
            for j in range(0, 13):
                data.append(f.readline().strip())
            data = list(map(float, " ".join(data).split()))

            X = np.array([data]) if X is None else np.append(X, [data], axis=0)

            y = np.array([f.readline().strip().split().index('0.80')]) if y is None else np.append(y, [f.readline().strip().split().index('0.80')], axis=0)

        y = np.array(y)

    return [X, y]


def normalize_data(data):
    for l in np.nditer(data, op_flags=['readwrite']):
        for e in np.nditer(l, op_flags=['readwrite']):
            e[...] = 0 if e == -0.5 else 1


if __name__ == '__main__':
    t = read_data('Trainingsdaten.pat')
    X_train = t[0]
    y_train = t[1]

    t = read_data('TestHandschriften.pat')
    X_test = t[0]
    y_test = t[1]

    # Gaussian
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    i = 0
    matches = 0
    for x in X_test:
        matches = matches + (1 if classifier.predict([x])[0] == (i % 26) else 0)
        i = i + 1

    print("%.4f%%" % (matches/i))

    # Bernoulli
    for alpha10 in range(1, 11):
        alpha = 0.1 * alpha10

        classifier = BernoulliNB(alpha=alpha, fit_prior=False)
        classifier.fit(X_train, y_train)

        i = 0
        matches = 0
        for x in X_test:
            matches = matches + (1 if classifier.predict([x])[0] == (i % 26) else 0)
            i = i + 1

        print("Bernoulli Alpha(%.2f):" % alpha)
        print("%.4f%%" % (matches/i))


    # Multinomial
    normalize_data(X_train)
    for alpha10 in range(1, 11):
        alpha = 0.1 * alpha10
        classifier = MultinomialNB(alpha=alpha, fit_prior=False)
        classifier.fit(X_train, y_train)

        i = 0
        matches = 0
        for x in X_test:
            matches = matches + (1 if classifier.predict([x])[0] == (i % 26) else 0)
            i = i + 1

        print("Multinomial Alpha(%.2f):" % alpha)
        print("%.4f%%" % (matches/i))
