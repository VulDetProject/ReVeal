import json
import numpy as np

from Vuld_SySe.representation_learning.representation_learning_api import RepresentationLearningModel
from Vuld_SySe.representation_learning.trainer import show_representation
from Vuld_SySe.representation_learning.tsne import plot_embedding


def read_data(path):
    features, targets = [], []
    json_data_file = open(path)
    data = json.load(json_data_file)
    np.random.shuffle(data)
    json_data_file.close()
    for d in data:
        features.append(d['graph_feature'])
        targets.append(d['target'])
    return np.array(features), np.array(targets)


ds = '../../data/after_ggnn/devign/v6/'
trx, trY = read_data(ds + 'train_GGNNinput_graph.json')
vals, valy = read_data(ds + 'valid_GGNNinput_graph.json')
tes, tey = read_data(ds + 'test_GGNNinput_graph.json')
print(len(tes[0]))
plot_embedding(tes, tey, 'tsnes/ggnn')
model = RepresentationLearningModel(print=True, max_patience=5, num_layers=1)
model.train(trx, trY)
for _x, _y in zip(tes, tey):
    model.dataset.add_data_entry(_x, _y, 'test')
show_representation(model.model, model.dataset.get_next_test_batch, model.dataset.initialize_test_batches(), 0,
                    "tsnes/representation-test")
ds = '../../data/full_experiment_real_data_processed/devign/full_graph/v1/graph_features/'
trx, trY = read_data(ds + 'train_GGNNinput_graph.json')
plot_embedding(tes, tey, 'tsnes/original')
