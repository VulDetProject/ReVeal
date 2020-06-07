import json
import numpy
import sys
from sklearn.model_selection import GridSearchCV

from representation_learning_api import RepresentationLearningModel

dataset = "chrome_debian"
# feature_name = sys.argv[2]
parts = ['train', 'valid']
features = []
targets = []

for part in parts:
    json_data_file = open(
        '../../data/full_experiment_real_data_processed/' + dataset + '/full_graph/v1/graph_features/' + part +
        '_GGNNinput_graph.json')
    data = json.load(json_data_file)
    json_data_file.close()
    for d in data:
        features.append(d['graph_feature'])
        targets.append(d['target'])
    del data
X = numpy.array(features)
Y = numpy.array(targets)
print('Dataset', X.shape, Y.shape, sep='\t', file=sys.stderr)

test_X, test_Y = [], []
json_data_file = open(
    '../../data/full_experiment_real_data_processed/' + dataset + '/full_graph/v1/graph_features/test_GGNNinput_graph.json')
data = json.load(json_data_file)
json_data_file.close()
for d in data:
    test_X.append(d['graph_feature'])
    test_Y.append(d['target'])
del data
test_X = numpy.array(test_X)
test_Y = numpy.array(test_Y)

clf = RepresentationLearningModel(print=False)

param_grid = {
    'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'lambda1': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'lambda2': [0, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2],
    'hidden_dim': [64, 128, 256, 512],
    'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'max_patience': [5, 10, 20]
}

search_params = GridSearchCV(estimator=clf, param_grid=param_grid, verbose=True, n_jobs=8)
search_params.fit(X, Y)

print(search_params.best_estimator_)
print('=' * 100)
print(search_params.best_params_)
print('=' * 100)
print(search_params.cv_results_)
print('=' * 100)
print(search_params.best_estimator_.evaluate(test_X, test_Y))
print('=' * 100)
print(search_params.best_estimator_.__dict__)
print('=' * 100)

