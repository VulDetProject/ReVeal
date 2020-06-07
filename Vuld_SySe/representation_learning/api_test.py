import argparse
import json
import numpy
import os
import sys
import torch
from representation_learning_api import RepresentationLearningModel
from sklearn.model_selection import train_test_split
from baseline_svm import SVMLearningAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='chrome_debian/balanced',
                        choices=['chrome_debian/balanced', 'chrome_debian/imbalanced', 'chrome_debian', 'devign'])
    parser.add_argument('--features', default='ggnn', choices=['ggnn', 'wo_ggnn'])
    parser.add_argument('--lambda1', default=0.5, type=float)
    parser.add_argument('--lambda2', default=0.001, type=float)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_balance', action='store_true')
    parser.add_argument('--baseline_model', default='svm')
    parser.add_argument('--num_layers', default=1, type=int)
    numpy.random.rand(1000)
    torch.manual_seed(1000)
    args = parser.parse_args()
    dataset = args.dataset
    feature_name = args.features
    parts = ['train', 'valid', 'test']
    if feature_name == 'ggnn':
        if dataset == 'chrome_debian/balanced':
            ds = '../../data/after_ggnn/chrome_debian/balance/v3/'
        elif dataset == 'chrome_debian/imbalanced':
            ds = '../../data/after_ggnn/chrome_debian/imbalance/v6/'
        elif dataset == 'devign':
            ds = '../../data/after_ggnn/devign/v6/'
        else:
            raise ValueError('Imvalid Dataset')
    else:
        if dataset == 'chrome_debian':
            ds = '../../data/full_experiment_real_data_processed/chrome_debian/full_graph/v1/graph_features/'
        elif dataset == 'devign':
            ds = '../../data/full_experiment_real_data_processed/devign/full_graph/v1/graph_features/'
        else:
            raise ValueError('Imvalid Dataset')
    assert isinstance(dataset, str)
    output_dir = 'results_test'
    if args.baseline:
        output_dir = 'baseline_' + args.baseline_model
        if args.baseline_balance:
            output_dir += '_balance'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file_name = output_dir + '/' + dataset.replace('/', '_') + '-' + feature_name + '-'
    if args.lambda1 == 0:
        assert args.lambda2 == 0
        output_file_name += 'cross-entropy-only-layers-'+ str(args.num_layers) + '.tsv'
    else:
        output_file_name += 'triplet-loss-layers-'+ str(args.num_layers) + '.tsv'
    output_file = open(output_file_name, 'w')
    features = []
    targets = []
    for part in parts:
        json_data_file = open(ds + part + '_GGNNinput_graph.json')
        data = json.load(json_data_file)
        json_data_file.close()
        for d in data:
            features.append(d['graph_feature'])
            targets.append(d['target'])
        del data
    X = numpy.array(features)
    Y = numpy.array(targets)
    print('Dataset', X.shape, Y.shape, numpy.sum(Y), sep='\t', file=sys.stderr)
    print('=' * 100, file=sys.stderr, flush=True)
    for _ in range(30):
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, sep='\t', file=sys.stderr, flush=True)
        if args.baseline:
            model = SVMLearningAPI(True, args.baseline_balance, model_type=args.baseline_model)
        else:
            model = RepresentationLearningModel(
                lambda1=args.lambda1, lambda2=args.lambda2, batch_size=128, print=True, max_patience=5, balance=True,
                num_layers=args.num_layers
            )
        model.train(train_X, train_Y)
        results = model.evaluate(test_X, test_Y)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t', flush=True,
              file=output_file)
        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t',
              file=sys.stderr, flush=True, end=('\n' + '=' * 100 + '\n'))
    output_file.close()
    pass
