import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1


def clone_analysis(data_paths):
    code = []
    labels = []
    positives = 0
    for file_name in data_paths:
        data = json.load(open(file_name))
        for example in data:
            code.append(example['tokenized'])
            l = 0
            if 'label' in example.keys():
                l = int(example['label'])
            elif 'lebel' in example.keys():
                l = int(example['lebel'])
            elif 'leble' in example.keys():
                l = int(example['leble'])
            elif 'lable' in example.keys():
                l = int(example['lable'])
            if l > 1:
                l = 1
            positives += l
            labels.append(l)
    print(len(code), len(labels), positives, len(labels) - positives)
    vectorizer = TfidfVectorizer(input=code, lowercase=False, ngram_range=(1, 3))
    X = vectorizer.fit_transform(code)
    model = KMeans(n_clusters=10, max_iter=100)
    model.fit(X)
    y = model.predict(X)
    cluster_to_positive = [0] * 10
    cluster_to_negative = [0] * 10
    for pred, label in zip(y, labels):
        if label == 1:
            cluster_to_positive[pred] += 1
        else:
            cluster_to_negative[pred] += 1
    print(cluster_to_positive)
    print(cluster_to_negative)
    percentages = [float(p) / (p + n) for p, n in zip(cluster_to_positive, cluster_to_negative)]
    for p in percentages:
        print(p)
    for _ in range(5):
        XTrain, XTest, YTrain, YTest = train_test_split(X, labels, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(XTrain, YTrain)
        predicted = model.predict(XTest)
        print('%.3f\t%.3f\t%.3f\t%.3f' % (
            acc(YTest, predicted) * 100, pr(YTest, predicted) * 100, rc(YTest, predicted) * 100, f1(YTest, predicted) * 100)
        )
    pass


if __name__ == '__main__':
    data = '../../data/draper/chrome_debian.json'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_source', help='Path of the json file of tha data', nargs='+',
        default=[
            data
        ]
    )
    args = parser.parse_args()
    clone_analysis(args.data_source)
    pass
