import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory():
    results = numpy.array([])
    targets = numpy.array([])

    with open('./Trainingsdaten.pat') as f:
        dim_x = int(f.readline().strip())
        dim_y = int(f.readline().strip())
        foo = int(f.readline().strip())
        target_dimensions = int(f.readline().strip())

        for j in range(260):
            data = ""
            for i in range(dim_y):
                # data.append([float(i) for i in f.readline().strip().split()])
                # data.append(list(map(float, f.readline().strip().split())))
                # data = data + list(map(float, f.readline().strip().split()))
                data = data + " " + f.readline().strip()

            # d = " ".join(data)
            # print(data)
            # target_vector = [float(i) for i in f.readline().strip().split()]
            target_vector = f.readline().strip()

            # print(d)
            # print(target_vector)
            # print(target_vector.index('0.80'))

            results = numpy.append(results, numpy.array(data.strip().split()))
            targets = numpy.append(targets, numpy.array([chr(int(int(target_vector.index('0.80')) / 6) + 97)]))
            # targets.append(chr(int(int(target_vector.index('0.80')) / 6) + 97))

            # result = result.append(tmp)
            # print(chr(int(int(target_vector.index('0.80')) / 6) + 97))
            # index.append(j)

    print(results)
    # return result
    # return DataFrame(result, index=index)
    # return DataFrame(result)
    return {'data': results, 'target': targets}



        #for line in f:
#    rows = []
#    index = []
#    for filename, message in readFiles(path):
#        rows.append({'message': message, 'class': classification})
#        index.append(filename)
#
#    return DataFrame(rows, index=index)

#data = DataFrame({'message': [], 'class': []})

#data = data.append(dataFrameFromDirectory('./emails/spam', 'spam'))
#data = data.append(dataFrameFromDirectory('./emails/ham', 'ham'))


if __name__ == '__main__':
    # data = DataFrame({'data': [], 'class': []})
    # data = data.append(dataFrameFromDirectory())
    data = dataFrameFromDirectory()

    # print(data)
    # vectorizer = CountVectorizer()
    # print(data['data'].values)
    # counts = vectorizer.fit_transform(data['data'].values)

    vectorizer = TfidfVectorizer(min_df=1,ngram_range=(1,2))
    # traindata = ['yes', 'yeah', 'i do not know', 'i am not sure', 'i have no idea', 'i'];
    X_train = vectorizer.fit_transform(data['data'])

    # Label Ids
    y_train = data['target']

    # Train classifier
    # clf.fit(X_train, y_train)

    print(X_train.size)
    print(y_train.size)

    # print(counts)
    classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    # classifier = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
    classifier.fit(X_train, y_train)

    examples = ['-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50',
                '-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
 0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50\
 0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50\
 0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50\
-0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50\
-0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50  0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50\
-0.50  0.50  0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50  0.50  0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50  0.50  0.50  0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50',
    '-0.50 -0.50 -0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50\
-0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50',
                '-0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50 -0.50\
-0.50 -0.50 -0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50 -0.50 -0.50']
    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print(predictions)
    #
    #
    # vectorizer = CountVectorizer()
    #
    # print(data[0]['data'])
    # # print(str(data[0]['data']))
    # counts = vectorizer.fit_transform(data[0]['data'])
    #
    # classifier = MultinomialNB()
    # targets = data[0]['class']
    # classifier.fit(counts, targets)
