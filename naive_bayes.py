# Naive Bayes

# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

import random
import time

TEST_SIZE_PRECISION = 10
ALPHA_PRECISION = 3

char_distribution = [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0, 6.3,
                     9.1, 2.6, 1.0, 2.3, 0.2, 2.0, 0.1]
normalized_distribution = [float(i) / max(char_distribution) for i in char_distribution]

STEP = int(sys.argv[1])

def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        ylim=None,
        cv=None,
        n_jobs=1):
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
    G_ALPHA = 0.1
    G_COLOR_1 = 'r'
    G_COLOR_2 = 'g'

    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        exploit_incremental_learning=True,
        n_jobs=n_jobs,
        train_sizes=np.linspace(.9, .1, TEST_SIZE_PRECISION),
        verbose=0
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=G_ALPHA,
        color=G_COLOR_1
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=G_ALPHA,
        color=G_COLOR_2
    )

    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color=G_COLOR_1,
        label="Training score"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color=G_COLOR_2,
        label="Cross-validation score"
    )

    plt.legend(loc="best")

    return plt


def init_plot(title, x_legend, y_legend):
    plt.title(title)

    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    plt.grid(True)


def plot(x, y, label):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)

    plt.plot(x, y, label=label)


def print_plot():
    plt.legend()
    plt.show()


def read_data(file_name):
    X = None
    y = None

    with open(file_name) as f:
        for l in f:
            # read data
            data = [l]
            for j in range(0, 13):
                data.append(f.readline().strip())
            data = list(map(float, " ".join(data).split()))

            X = np.array([data]) if X is None else np.append(X, [data], axis=0)

            y = np.array([f.readline().strip().split().index('0.80')]) if y is None else np.append(y, [
                f.readline().strip().split().index('0.80')], axis=0)

        y = np.array(y)

    return [X, y]


def normalize_data(data):
    _data = np.array(data, copy=True)

    for l in np.nditer(_data, op_flags=['readwrite']):
        for e in np.nditer(l, op_flags=['readwrite']):
            e[...] = 0 if e == -0.5 else 1

    return _data


def get_distributed_subset(data, target):
    index_list = [
        [e for e in range(0, len(data), 26) if e == 1 or random.random() < normalized_distribution[0]],
        [e for e in range(1, len(data), 26) if e == 1 or random.random() < normalized_distribution[1]],
        [e for e in range(2, len(data), 26) if e == 2 or random.random() < normalized_distribution[2]],
        [e for e in range(3, len(data), 26) if e == 3 or random.random() < normalized_distribution[3]],
        [e for e in range(4, len(data), 26) if e == 4 or random.random() < normalized_distribution[4]],
        [e for e in range(5, len(data), 26) if e == 5 or random.random() < normalized_distribution[5]],
        [e for e in range(6, len(data), 26) if e == 6 or random.random() < normalized_distribution[6]],
        [e for e in range(7, len(data), 26) if e == 7 or random.random() < normalized_distribution[7]],
        [e for e in range(8, len(data), 26) if e == 8 or random.random() < normalized_distribution[8]],
        [e for e in range(9, len(data), 26) if e == 9 or random.random() < normalized_distribution[9]],
        [e for e in range(10, len(data), 26) if e == 10 or random.random() < normalized_distribution[10]],
        [e for e in range(11, len(data), 26) if e == 11 or random.random() < normalized_distribution[11]],
        [e for e in range(12, len(data), 26) if e == 12 or random.random() < normalized_distribution[12]],
        [e for e in range(13, len(data), 26) if e == 13 or random.random() < normalized_distribution[13]],
        [e for e in range(14, len(data), 26) if e == 14 or random.random() < normalized_distribution[14]],
        [e for e in range(15, len(data), 26) if e == 15 or random.random() < normalized_distribution[15]],
        [e for e in range(16, len(data), 26) if e == 16 or random.random() < normalized_distribution[16]],
        [e for e in range(17, len(data), 26) if e == 17 or random.random() < normalized_distribution[17]],
        [e for e in range(18, len(data), 26) if e == 18 or random.random() < normalized_distribution[18]],
        [e for e in range(19, len(data), 26) if e == 19 or random.random() < normalized_distribution[19]],
        [e for e in range(20, len(data), 26) if e == 20 or random.random() < normalized_distribution[20]],
        [e for e in range(21, len(data), 26) if e == 21 or random.random() < normalized_distribution[21]],
        [e for e in range(22, len(data), 26) if e == 22 or random.random() < normalized_distribution[22]],
        [e for e in range(23, len(data), 26) if e == 23 or random.random() < normalized_distribution[23]],
        [e for e in range(24, len(data), 26) if e == 24 or random.random() < normalized_distribution[24]],
        [e for e in range(25, len(data), 26) if e == 25 or random.random() < normalized_distribution[25]]
    ]
    index_list = [item for sublist in index_list for item in sublist]

    _data = [e for j, e in enumerate(data) if j in index_list]
    _target = [e for j, e in enumerate(target) if j in index_list]

    return np.array(_data), np.array(_target)


def my_train_test_split(X, y, test_size, random=True, is_relative_size=True):
    if is_relative_size:
        abs_test_size = round(test_size * len(X))
    else:
        abs_test_size = test_size
        test_size = abs_test_size / len(X)

    if random:
        head_X = X[:52]
        tail_X = X[52:]
        head_y = y[:52]
        tail_y = y[52:]

        X_train, X_test, y_train, y_test = train_test_split(tail_X, tail_y, test_size=test_size)

        X_train = np.append(X_train, head_X[26:], axis=0)
        y_train = np.append(y_train, head_y[26:], axis=0)

        X_test = np.append(X_test, head_X[:26], axis=0)
        y_test = np.append(y_test, head_y[:26], axis=0)
    else:
        X_train = np.array(X[abs_test_size:])
        X_test  = np.array(X[:abs_test_size])
        y_train = np.array(y[abs_test_size:])
        y_test  = np.array(y[:abs_test_size])

    return X_train, X_test, y_train, y_test


def get_priors(y):
    result = [0] * 26

    for e in y:
        result[e] += 1

    return [e / sum(result) for e in result]


if __name__ == '__main__':
    X, y = read_data('data.pat')

    if STEP == 1:
        best_gaussian = {
            'training_size': 0,
            'score': 0.0,
        }
        best_bernoulli = {
            'training_size': 0,
            'score': 0.0,
            'alpha': 0.0,
        }
        best_multinomial = {
            'training_size': 0,
            'score': 0.0,
            'alpha': 0.0,
        }

        ####################################################################################################################
        # Gauss
        ####################################################################################################################
        gaussian_accuracy = []
        gaussian_training_duration = []
        for test_size in np.linspace(.9, .1, TEST_SIZE_PRECISION):
            # init
            X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)
            training_size = len(y_train)

            # classifier
            # cGauss = GaussianNB(priors=get_priors(y_train))
            cGauss = GaussianNB()

            # train
            t1 = time.time()
            cGauss.fit(X_train, y_train)
            training_duration = time.time() - t1

            # predict
            tmp = []
            for e in X_test:
                t1 = time.time()
                cGauss.predict(e.reshape(1, -1))
                tmp.append(time.time() - t1)
            prediction_duration = sum(tmp) / len(tmp)

            # score
            score = cGauss.score(X_test, y_test)
            if score > best_gaussian['score']:
                best_gaussian['score'] = score
                best_gaussian['training_size'] = training_size

            gaussian_accuracy.append((training_size, score))
            gaussian_training_duration.append((training_size, training_duration))

        ################################################################################################################
        # Bernoulli
        ################################################################################################################
        bernoulli_accuracy = []
        bernoulli_training_duration = []
        for alpha in np.linspace(.1, 1, ALPHA_PRECISION):
            for test_size in np.linspace(.9, .1, TEST_SIZE_PRECISION):
                # init
                X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)
                training_size = len(y_train)

                # classifier
                cBernoulli = BernoulliNB(alpha=alpha, fit_prior=False, class_prior=get_priors(y_test))

                # train
                t1 = time.time()
                cBernoulli.fit(X_train, y_train)
                training_duration = time.time() - t1

                # predict
                tmp = []
                for e in X_test:
                    t1 = time.time()
                    cBernoulli.predict(e.reshape(1, -1))
                    tmp.append(time.time() - t1)
                prediction_duration = sum(tmp) / len(tmp)

                # score
                score = cBernoulli.score(X_test, y_test)
                if score > best_bernoulli['score']:
                    best_bernoulli['score'] = score
                    best_bernoulli['alpha'] = alpha
                    best_bernoulli['training_size'] = training_size

                bernoulli_accuracy.append((training_size, alpha, score))
                bernoulli_training_duration.append((training_size, alpha, training_duration))

        ################################################################################################################
        # Multinomial
        ################################################################################################################
        multinomial_accuracy = []
        multinomial_training_duration = []
        for alpha in np.linspace(.1, 1, ALPHA_PRECISION):
            for test_size in np.linspace(.9, .1, TEST_SIZE_PRECISION):
                # init
                X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)
                training_size = len(y_train)

                # classifier
                cMultinomial = MultinomialNB(alpha=alpha, fit_prior=False, class_prior=get_priors(y_test))

                # train
                t1 = time.time()
                X_train = normalize_data(X_train)
                cMultinomial.fit(X_train, y_train)
                training_duration = time.time() - t1

                # predict
                tmp = []
                for e in normalize_data(X_test):
                    t1 = time.time()
                    cMultinomial.predict(e.reshape(1, -1))
                    tmp.append(time.time() - t1)
                prediction_duration = sum(tmp) / len(tmp)

                # score
                score = cMultinomial.score(normalize_data(X_test), y_test)
                if score > best_multinomial['score']:
                    best_multinomial['score'] = score
                    best_multinomial['alpha'] = alpha
                    best_multinomial['training_size'] = training_size

                multinomial_accuracy.append((training_size, alpha, score))
                multinomial_training_duration.append((training_size, alpha, training_duration))

        # debug
        print(best_gaussian)
        print(best_bernoulli)
        print(best_multinomial)

        # Plot graphs
        training_set_sizes = [e * len(X) for e in np.linspace(.1, .9, TEST_SIZE_PRECISION)]

        init_plot('Accuracy as a function of training sets', 'Training Sets', 'Accuracy')
        plot(training_set_sizes, [e[1] for e in gaussian_accuracy], 'Gauss')
        plot(training_set_sizes, [e[2] for e in bernoulli_accuracy if e[1] == best_bernoulli['alpha']], 'Bernoulli')
        plot(training_set_sizes, [e[2] for e in multinomial_accuracy if e[1] == best_multinomial['alpha']], 'Multinomial')
        print_plot()

        init_plot('Learning duration as a function of training sets', 'Training Sets', 'Duration')
        plot(training_set_sizes, [e[1] for e in gaussian_training_duration], 'Gauss')
        plot(training_set_sizes, [e[2] for e in bernoulli_training_duration if e[1] == best_bernoulli['alpha']], 'Multinomial')
        plot(training_set_sizes, [e[2] for e in multinomial_training_duration if e[1] == best_multinomial['alpha']], 'Bernoulli')
        print_plot()

        plot_learning_curve(GaussianNB(), 'Gauss', X, y).show()
        plot_learning_curve(BernoulliNB(binarize=0.0, alpha=best_bernoulli['alpha']), 'BernoulliNB', X, y).show()
        plot_learning_curve(MultinomialNB(alpha=best_multinomial['alpha']), 'MultinomialNB', normalize_data(X), y).show()

    elif STEP == 2:
        ################################################################################################################
        # Multinomial - Priors
        ################################################################################################################
        cMultinomial = MultinomialNB(alpha=0.1, fit_prior=False, class_prior=[1/26] * 26)
        data_p = []
        data_np = []
        for i in range(0, round(len(X) / 26) - 1):
            training_size_absolute = 26 * (i + 1)

            cMultinomialP = MultinomialNB(alpha=0.2, fit_prior=False, class_prior=normalized_distribution)
            cMultinomialNP = MultinomialNB(alpha=0.2, fit_prior=False, class_prior=[1/26] * 26)

            training_size_absolute = 26 * (i + 1)

            X_train = X[:training_size_absolute]
            X_test = X[training_size_absolute:]
            y_train = y[:training_size_absolute]
            y_test = y[training_size_absolute:]

            cMultinomialP.fit(normalize_data(X_train), y_train)
            cMultinomialNP.fit(normalize_data(X_train), y_train)

            X_test, y_test = get_distributed_subset(X_test, y_test)

            data_p.append((i + 1, cMultinomialP.score(normalize_data(X_test), y_test)))
            data_np.append((i + 1, cMultinomialNP.score(normalize_data(X_test), y_test)))

        # debug
        print(data_p)
        print(data_np)

        # Plot graphs
        init_plot('Accuracy as a function of learned training sets', 'Training Sets', 'Accuracy')
        plot([e[0] for e in data_p], [e[1] for e in data_p], 'With Priors')
        plot([e[0] for e in data_np], [e[1] for e in data_np], 'Without Priors')
        print_plot()

    elif STEP == 3:
        ################################################################################################################
        # Multinomial - Aufgabenstellung Epochen
        ################################################################################################################
        cMultinomial = MultinomialNB(alpha=0.1, fit_prior=False, class_prior=[1/26] * 26)
        data_epo = []
        for i in range(0, round(len(X) / 26) - 1):
            X_train = X[26 * i:][:26]
            X_test = X[26 * (i + 1):]
            y_train = y[26 * i:][:26]
            y_test = y[26 * (i + 1):]

            cMultinomial.partial_fit(normalize_data(X_train), y_train, classes=range(0, 26))

            data_epo.append((i + 1, cMultinomial.score(normalize_data(X_test), y_test)))

        data_full = []
        for i in range(0, round(len(X) / 26) - 1):
            cMultinomial = MultinomialNB(alpha=0.1, fit_prior=False, class_prior=[1/26] * 26)
            training_size_absolute = 26 * (i + 1)

            X_train = X[:training_size_absolute]
            X_test = X[training_size_absolute:]
            y_train = y[:training_size_absolute]
            y_test = y[training_size_absolute:]

            cMultinomial.fit(normalize_data(X_train), y_train)

            data_full.append((i + 1, cMultinomial.score(normalize_data(X_test), y_test)))

        # debug
        print(data_epo)
        print(data_full)

        # Plot graphs
        init_plot('Accuracy as a function of learned training sets', 'Training Sets', 'Accuracy')
        plot([e[0] for e in data_epo], [e[1] for e in data_epo], 'Incremental Learning')
        plot([e[0] for e in data_full], [e[1] for e in data_full], 'Full Learning')
        print_plot()
