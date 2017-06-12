import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
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
    result = []
    index = []

    with open('./Trainingsdaten.pat') as f:
        dim_x = int(f.readline().strip())
        dim_y = int(f.readline().strip())
        foo = int(f.readline().strip())
        target_dimensions = int(f.readline().strip())

        for j in range(260):
            data = []
            for i in range(dim_y):
                # data.append([float(i) for i in f.readline().strip().split()])
                data.append(f.readline().strip())

            d = "\n".join(data)
            # target_vector = [float(i) for i in f.readline().strip().split()]
            target_vector = f.readline().strip()
            result.append({'data': d, 'class': chr(int(int(target_vector.index('0.80')) / 6) + 97)})
            print(chr(int(int(target_vector.index('0.80')) / 6) + 97))
            index.append(j)

    # return result
    # return DataFrame(result, index=index)
    return DataFrame(result)



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
    data = DataFrame({'data': [], 'class': []})

    data = data.append(dataFrameFromDirectory())

    print(data)
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data['data'].values)

    # print(counts)
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)

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
-0.50 -0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50 -0.50 -0.50']
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
