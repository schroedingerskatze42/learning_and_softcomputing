# Naive Bayes

# Importing the libraries
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import random
import time

char_derivation = [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.2, 0.8, 4.0, 2.4, 6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1,
                   2.6, 1.0, 2.3, 0.2, 2.0, 0.1]
normalized_derivation = [float(i) / max(char_derivation) for i in char_derivation]
normalized_derivation_1 = [float(i) / sum(char_derivation) for i in char_derivation]



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1, 10)):
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
        # cv=cv,
        cv=11,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
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


def get_derivated_subset(data, target):
    index_list = [
        [e for e in range(0, len(data), 26) if e == 1 or random.random() < normalized_derivation[0]],
        [e for e in range(1, len(data), 26) if e == 1 or random.random() < normalized_derivation[1]],
        [e for e in range(2, len(data), 26) if e == 2 or random.random() < normalized_derivation[2]],
        [e for e in range(3, len(data), 26) if e == 3 or random.random() < normalized_derivation[3]],
        [e for e in range(4, len(data), 26) if e == 4 or random.random() < normalized_derivation[4]],
        [e for e in range(5, len(data), 26) if e == 5 or random.random() < normalized_derivation[5]],
        [e for e in range(6, len(data), 26) if e == 6 or random.random() < normalized_derivation[6]],
        [e for e in range(7, len(data), 26) if e == 7 or random.random() < normalized_derivation[7]],
        [e for e in range(8, len(data), 26) if e == 8 or random.random() < normalized_derivation[8]],
        [e for e in range(9, len(data), 26) if e == 9 or random.random() < normalized_derivation[9]],
        [e for e in range(10, len(data), 26) if e == 10 or random.random() < normalized_derivation[10]],
        [e for e in range(11, len(data), 26) if e == 11 or random.random() < normalized_derivation[11]],
        [e for e in range(12, len(data), 26) if e == 12 or random.random() < normalized_derivation[12]],
        [e for e in range(13, len(data), 26) if e == 13 or random.random() < normalized_derivation[13]],
        [e for e in range(14, len(data), 26) if e == 14 or random.random() < normalized_derivation[14]],
        [e for e in range(15, len(data), 26) if e == 15 or random.random() < normalized_derivation[15]],
        [e for e in range(16, len(data), 26) if e == 16 or random.random() < normalized_derivation[16]],
        [e for e in range(17, len(data), 26) if e == 17 or random.random() < normalized_derivation[17]],
        [e for e in range(18, len(data), 26) if e == 18 or random.random() < normalized_derivation[18]],
        [e for e in range(19, len(data), 26) if e == 19 or random.random() < normalized_derivation[19]],
        [e for e in range(20, len(data), 26) if e == 20 or random.random() < normalized_derivation[20]],
        [e for e in range(21, len(data), 26) if e == 21 or random.random() < normalized_derivation[21]],
        [e for e in range(22, len(data), 26) if e == 22 or random.random() < normalized_derivation[22]],
        [e for e in range(23, len(data), 26) if e == 23 or random.random() < normalized_derivation[23]],
        [e for e in range(24, len(data), 26) if e == 24 or random.random() < normalized_derivation[24]],
        [e for e in range(25, len(data), 26) if e == 25 or random.random() < normalized_derivation[25]]
    ]
    index_list = [item for sublist in index_list for item in sublist]
    # print(flat_list)
    # exit(1)
    _data = [e for j, e in enumerate(data) if j in index_list]
    _target = [e for j, e in enumerate(target) if j in index_list]

    # print(_data)
    # print(len(_data))
    # print(_target)
    # print(len(_target))
    # exit(1)
    return {'data': np.array(_data), 'target': np.array(_target)}


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


def my_train_test_split(X, y, test_size):
    head_X = X[:52]
    tail_X = X[52:]
    head_y = y[:52]
    tail_y = y[52:]

    X_train, X_test, y_train, y_test = train_test_split(tail_X, tail_y, test_size=test_size)

    X_train = np.append(X_train, head_X[26:], axis=0)
    y_train = np.append(y_train, head_y[26:], axis=0)

    X_test = np.append(X_train, head_X[:26], axis=0)
    y_test = np.append(y_train, head_y[:26], axis=0)

    return X_train, X_test, y_train, y_test


def inverse_priors(priors):
    return [1/26 - (e - 1/26) for e in priors]


def get_priors(y):
    result = [0] * 26

    for e in y:
        result[e] += 1

    return [e / sum(result) for e in result]


if __name__ == '__main__':
    X, y = read_data('data.pat')

    best_gaussian = {
        'test_size': 0,
        'score': 0.0,
    }
    best_bernoulli = {
        'test_size': 0,
        'score': 0.0,
        'alpha': 0.0,
    }
    best_multinomial = {
        'test_size': 0,
        'score': 0.0,
        'alpha': 0.0,
    }
    #
    # gaussian_accuracy = []
    # gaussian_training_duration = []
    # gaussian_prediction_duration = []
    # for d in range(1, 10):
    #     test_size = 1 - (d / 10)
    #
    #     X_train, X_test, y_train, y_test = my_train_test_split(normalize_data(X), y, test_size=test_size)  # todo write own
    #
    #     priors = get_priors(y_test)
    #
    #     # only way it makes sense in our context
    #     # cGauss = GaussianNB(priors=get_priors(y_train))
    #     cGauss = GaussianNB()
    #
    #     t1 = time.time()
    #     cGauss.fit(X_train, y_train)
    #     training_duration = time.time() - t1
    #     # cGauss.set_params(priors=get_priors(y_train))
    #
    #     tmp = []
    #     for e in X_test:
    #         t1 = time.time()
    #         cGauss.predict(e.reshape(1, -1))
    #         tmp.append(time.time() - t1)
    #     prediction_duration = sum(tmp) / len(tmp)
    #
    #     gaussian_accuracy.append((round((1 - test_size) * len(X_train)), cGauss.score(X_test, y_test)))
    #     gaussian_training_duration.append((round((1 - test_size) * len(X_train)), training_duration))
    #     gaussian_prediction_duration.append((round((1 - test_size) * len(X_train)), prediction_duration))

    ###########
    # Bernoulli
    bernoulli_accuracy = []
    bernoulli_training_duration = []
    bernoulli_prediction_duration = []

    PRECISION = 10
    plot_X = np.linspace(.1, 1, PRECISION)
    plot_Y = np.linspace(.1, 1, PRECISION)
    plot_Z = []
    for alpha in np.linspace(.1, 1, PRECISION):
        _Z = []
        for d in np.linspace(.1, 0.9, PRECISION):
            test_size = 1 - d

            X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)

            cBernoulli = BernoulliNB(binarize=0.0, alpha=alpha, fit_prior=False, class_prior=get_priors(y_test))

            t1 = time.time()
            cBernoulli.fit(X_train, y_train)
            training_duration = time.time() - t1

            tmp = []
            for e in X_test:
                t1 = time.time()
                cBernoulli.predict(e.reshape(1, -1))
                tmp.append(time.time() - t1)

            _Z.append(cBernoulli.score(X_test, y_test))

            bernoulli_accuracy.append(((1 - test_size) * len(X_train), alpha, cBernoulli.score(X_test, y_test)))
            bernoulli_training_duration.append(((1 - test_size) * len(X_train), alpha, training_duration))
            bernoulli_prediction_duration.append((round((1 - test_size) * len(X_train)), alpha, sum(tmp) / len(tmp)))#

        plot_Z.append(_Z)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel('Test Size')
    # ax.set_ylabel('Alpha')
    # ax.set_zlabel('Accuracy')
    # ax.plot_surface(np.unique(plot_X), np.unique(plot_Y), plot_Z)
    # plt.show()
    #
    print(bernoulli_accuracy)

    # print([e[1] for i, e in enumerate(bernoulli_accuracy) if i % PRECISION == 0])
    # print([e[2] for i, e in enumerate(bernoulli_accuracy) if i % PRECISION == 0])
    # init_plot('Prediction accuracy as a function of alpha', 'Alpha', 'Accuracy')
    # plot([e[1] for i, e in enumerate(bernoulli_accuracy) if i % PRECISION == 0], [e[2] for i, e in enumerate(bernoulli_accuracy) if i % PRECISION == 0], 'Bernoulli')
    # plot([e[0] for e in multinomial_prediction_duration], [e[1] for e in multinomial_prediction_duration], 'Multinomial')
    # plot([e[0] for e in bernoulli_prediction_duration], [e[1] for e in bernoulli_prediction_duration], 'Bernoulli')
    # print_plot()
    exit(1)

    for d in range(1, 10):
        test_size = 1 - (d / 10)

        X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)

        X_train = np.append(X_train, X[:26], axis=0)
        y_train = np.append(y_train, y[:26], axis=0)
        cBernoulli = BernoulliNB(binarize=0.0, alpha=1.0, fit_prior=False, class_prior=get_priors(y_test))

        t1 = time.time()
        cBernoulli.fit(X_train, y_train)
        training_duration = time.time() - t1

        tmp = []
        for e in X_test:
            t1 = time.time()
            cBernoulli.predict(e.reshape(1, -1))
            tmp.append(time.time() - t1)

        print(cBernoulli.score(X_test, y_test))

        bernoulli_accuracy.append(((1 - test_size) * len(X_train), cBernoulli.score(X_test, y_test)))
        bernoulli_training_duration.append(((1 - test_size) * len(X_train), training_duration))
        bernoulli_prediction_duration.append((round((1 - test_size) * len(X_train)), sum(tmp) / len(tmp)))
    multinomial_accuracy = []
    multinomial_training_duration = []
    multinomial_prediction_duration = []
    for d in range(1, 10):
        test_size = 1 - (d / 10)

        X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=test_size)

        cMulti = MultinomialNB()

        t1 = time.time()
        X_train = normalize_data(X_train)
        cMulti.fit(X_train, y_train)
        training_duration = time.time() - t1

        tmp = []
        for e in normalize_data(X_test):
            t1 = time.time()
            cMulti.predict(e.reshape(1, -1))
            tmp.append(time.time() - t1)

        multinomial_accuracy.append(((1 - test_size) * len(X_train), cMulti.score(normalize_data(X_test), y_test)))
        multinomial_training_duration.append(((1 - test_size) * len(X_train), training_duration))
        multinomial_prediction_duration.append((round((1 - test_size) * len(X_train)), sum(tmp) / len(tmp)))

    plot_learning_curve(BernoulliNB(), 'BernoulliNB', X, y).show()
    plot_learning_curve(MultinomialNB(), 'MultinomialNB', normalize_data(X), y).show()
    plot_learning_curve(GaussianNB(priors=[1/26] * 26), 'Gauss', X, y).show()

    # init_plot('Accuracy as a function of Training Sets', 'Training Sets', 'Accuracy')
    # plot([e[0] for e in gaussian_accuracy], [e[1] for e in gaussian_accuracy], 'Gauss')
    # plot([e[0] for e in multinomial_accuracy], [e[1] for e in multinomial_accuracy], 'Multinomial')
    # plot([e[0] for e in bernoulli_accuracy], [e[1] for e in bernoulli_accuracy], 'Bernoulli')
    # print_plot()
    #
    # init_plot('Learning duration as a function of Training Sets', 'Training Sets', 'Duration')
    # plot([e[0] for e in gaussian_training_duration], [e[1] for e in gaussian_training_duration], 'Gauss')
    # plot([e[0] for e in multinomial_training_duration], [e[1] for e in multinomial_training_duration], 'Multinomial')
    # plot([e[0] for e in bernoulli_training_duration], [e[1] for e in bernoulli_training_duration], 'Bernoulli')
    # print_plot()
    #
    # init_plot('Prediction duration as a function of Training Sets', 'Training Sets', 'Duration')
    # plot([e[0] for e in gaussian_prediction_duration], [e[1] for e in gaussian_prediction_duration], 'Gauss')
    # plot([e[0] for e in multinomial_prediction_duration], [e[1] for e in multinomial_prediction_duration], 'Multinomial')
    # plot([e[0] for e in bernoulli_prediction_duration], [e[1] for e in bernoulli_prediction_duration], 'Bernoulli')
    # print_plot()

    exit(1)

    best_gaussian = {
        'test_size': 0,
        'score': 0.0,
    }
    best_bernoulli = {
        'test_size': 0,
        'score': 0.0,
        'alpha': 0.0,
    }
    best_multinomial = {
        'test_size': 0,
        'score': 0.0,
        'alpha': 0.0,
    }

    # for d in range(1, 10):
    for d in range(1):
        test_size = 0.2
        # test_size = d / 10
        X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=test_size)

        print("Testing with Test Size %.2f" % test_size)

        ##########
        # Gaussian
        # classifier = GaussianNB()
        # char_derivation = [float(e) / 100 for e in char_derivation]
        print([round(float(i) / sum(char_derivation), 4) for i in char_derivation])
        print(sum([round(float(i) / sum(char_derivation), 4) for i in char_derivation]))
        # classifier = GaussianNB(priors=[round(float(i) / sum(char_derivation), 4) for i in char_derivation])
        classifier = GaussianNB(priors=[1/26]*26)
        classifier = GaussianNB()

        classifier.fit(X_train, y_train)
        print(classifier.class_prior_)

        score = classifier.score(X_test, y_test)
        if score > best_gaussian['score']:
            best_gaussian['test_size'] = test_size
            best_gaussian['score'] = score

        print(best_gaussian)
        exit(1)
        ###########
        # Bernoulli
        for a in range(1, 11):
            alpha = a / 10

            classifier = BernoulliNB(alpha=alpha, fit_prior=False)
            classifier.fit(X_train, y_train)

            score = classifier.score(X_test, y_test)

            if score > best_bernoulli['score']:
                best_bernoulli['test_size'] = test_size
                best_bernoulli['score'] = score
                best_bernoulli['alpha'] = alpha

        #############
        # Multinomial
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        for a in range(1, 11):
            alpha = a / 10

            classifier = MultinomialNB(alpha=alpha, fit_prior=False)
            classifier.fit(X_train, y_train)

            score = classifier.score(X_test, y_test)

            if score > best_multinomial['score']:
                best_multinomial['test_size'] = test_size
                best_multinomial['score'] = score
                best_multinomial['alpha'] = alpha

    t = read_data('Daten.pat')
    X = t[0]
    y = t[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # classifier = GaussianNB()
    # title = "Best Gaussian (Test Size: " + str(best_gaussian['test_size']) + ")"
    # plot_learning_curve(classifier, title, X, y)
    # plt.show()
    #
    # X = t[0]
    # y = t[1]
    # classifier = BernoulliNB(alpha=best_bernoulli['alpha'], fit_prior=False)
    # title = "Best Bernoulli (Test Size: " + str(best_bernoulli['test_size']) + ", Alpha: " + str(
    #     best_bernoulli['alpha']) + ")"
    # plot_learning_curve(classifier, title, X, y)
    # plt.show()
    #
    # classifier = BernoulliNB(alpha=best_bernoulli['alpha'], fit_prior=True, class_prior=normalized_derivation)
    # X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=test_size)
    # classifier.fit(X_train, y_train)
    # foo = get_derivated_subset(X_test, y_test)
    # print(foo)
    # print(classifier.score(foo['data'], foo['target']))
    #
    # # X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=best_multinomial['test_size'])
    # X = normalize_data(t[0])
    # y = t[1]
    # classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=False)
    # title = "Best Multinomial (Test Size: " + str(best_multinomial['test_size']) + ", Alpha: " + str(
    #     best_multinomial['alpha']) + ")"
    # plot_learning_curve(classifier, title, X, y)
    # plt.show()

    classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=True, class_prior=normalized_derivation)
    classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=False)
    # X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=test_size)
    # X_train = normalize_data(t[0])[:26]
    # y_train = t[1][:26]
    # X_train = normalize_data(t[0])[:260]
    # y_train = t[1][:260]
    X_train = normalize_data(t[0])[:520]
    y_train = t[1][:520]
    X_test = normalize_data(t[0])[520:]
    y_test = t[1][520:]
    classifier.fit(X_train, y_train)
    # foo = get_derivated_subset(X_test, y_test)
    # print(classifier.score(foo['data'], foo['target']))
    print(classifier.predict(X_test))

    # # foo = get_derivated_subset(t[0], t[1])
    # X_train, X_test, y_train, y_test = train_test_split(t[0], t[1], test_size=0.2)
    # # foo = (t[0], t[1])
    # # X = foo['data']
    # # y = foo['target']
    # X = X_train
    # y = y_train
    # # y = t[1]
    #
    # print(type(X))
    # print(len(X))
    # print(type(y))
    # print(len(y))
    # # print(X)
    # normalize_data(X)
    # # print(normalized_derivation)
    #
    # # classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=False)
    # # classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=False, class_prior=derivation)
    # # classifier = MultinomialNB(alpha=best_multinomial['alpha'], fit_prior=False, class_prior=normalized_derivation)
    # classifier = MultinomialNB(alpha=0.2, fit_prior=True, class_prior=normalized_derivation)
    # classifier.fit(X, y)
    #
    # y_pred = classifier.predict(X)
    # print(y_pred)
    # print(y)
    #
    # foo = get_derivated_subset(t[0], t[1])
    # X = foo['data']
    # y = foo['target']
    # # X = t[0]
    # # y = t[1]
    # print(classifier.class_prior)
    # score = classifier.score(X, y)
    # print(score)

    print(best_gaussian)
    print(best_bernoulli)
    print(best_multinomial)
