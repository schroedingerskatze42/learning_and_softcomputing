# Naive Bayes

# Importing the libraries
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



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
    t = read_data('Daten.pat')

    best_gaussian = {
        # 'i': 0,
        # 'matches': 0,
        'test_size': 0,
        # 'result': 0.0,
        'score': 0.0,
    }
    best_bernoulli = {
        # 'i': 0,
        # 'matches': 0,
        'test_size': 0,
        # 'result': 0.0,
        'score': 0.0,
        'alpha': 0.0,
    }
    best_multinomial = {
        # 'i': 0,
        # 'matches': 0,
        'test_size': 0,
        # 'result': 0.0,
        'score': 0.0,
        'alpha': 0.0,
    }
# plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)


    for d in range(1):
        # test_size = float((1 + d * 5) / 100)
        test_size = 0.4
        print("Testing with Test Size %.2f" % test_size)

        X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=test_size)

        # Gaussian
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        i = 0
        # matches = 0
        # for x in X_test:
        #     matches = matches + (1 if classifier.predict([x])[0] == y_test[i] else 0)
        #     i = i + 1

        # result = float(100 * (matches / i))
        score = classifier.score(X_test, y_test)
        if score > best_gaussian['score']:
            best_gaussian['i'] = i
            # best_gaussian['matches'] = matches
            best_gaussian['test_size'] = test_size
            best_gaussian['score'] = score
            # best_gaussian['result'] = float(100 * matches / i)
            # print(best_gaussian)

        # print("GaussianNB %d:" % test_size)
        # print("%.4f%%" % (100 * matches / i))
        #
        # Bernoulli
        for alpha10 in range(1, 11):
            alpha = 0.1 * alpha10

            classifier = BernoulliNB(alpha=alpha, fit_prior=False)
            classifier.fit(X_train, y_train)

            i = 0
            # matches = 0
            # for x in X_test:
            #     matches = matches + (1 if classifier.predict([x])[0] == y_test[i] else 0)
            #     i = i + 1
            score = classifier.score(X_test, y_test)

            if score > best_bernoulli['score']:
                best_bernoulli['i'] = i
                # best_bernoulli['matches'] = matches
                best_bernoulli['test_size'] = test_size
                best_bernoulli['score'] = score
                best_bernoulli['alpha'] = alpha
                # best_bernoulli['result'] = float(100 * matches / i)#

            # plot_learning_curve(classifier, "foo", X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
            plot_learning_curve(classifier, "foo", X_test, y_test)
            plt.show()


            # print("Bernoulli Alpha(%.2f), TestSize(%.2f):" % (alpha, test_size))
            # print("%.4f%%" % (100*matches/i))


        # # Multinomial
        normalize_data(X_train)
        for alpha10 in range(1, 11):
            alpha = 0.1 * alpha10
            classifier = MultinomialNB(alpha=alpha, fit_prior=False)
            classifier.fit(X_train, y_train)

            i = 0
            # matches = 0
            # for x in X_test:
            #     matches = matches + (1 if classifier.predict([x])[0] == y_test[i] else 0)
            #     i = i + 1
            score = classifier.score(X_test, y_test)

            if score > best_multinomial['score']:
                best_multinomial['i'] = i
                # best_multinomial['matches'] = matches
                best_multinomial['test_size'] = test_size
                best_multinomial['score'] = score
                best_multinomial['alpha'] = alpha
                # best_multinomial['result'] = float(100 * matches / i)

            # print("Multinomial Alpha(%.2f), TestSize(%.2f):" % (alpha, test_size))
            # print("%.4f%%" % (100*matches/i))

    print(best_gaussian)
    print(best_bernoulli)
    print(best_multinomial)
