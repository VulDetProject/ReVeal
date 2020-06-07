import os
import sys
from torch.optim import Adam

sys.path.append(os.path.abspath(__file__))

from graph_dataset import create_dataset
from models import MetricLearningModel
from trainer import train, show_representation
import numpy as np
import random
import torch
import warnings

warnings.filterwarnings('ignore')
import argparse
from tsne import plot_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', help='Base Dir',
        default='/home/saikatc/DATA/CCS-Vul_Det/data/'
                'full_experiment_real_data_processed/chrome_debian/full_graph/v1/graph_data'
    )
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    np.random.seed(1000)
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    random.seed(1000)
    base_dir = args.base_dir
    dataset = create_dataset(
        train_file=base_dir + '/train_GGNNinput_graph.json',
        valid_file=base_dir + '/valid_GGNNinput_graph.json',
        test_file=base_dir + '/test_GGNNinput_graph.json',
        batch_size=64,
        output_buffer=sys.stderr
    )
    num_epochs = 200
    dataset.initialize_dataset(balance=True)
    train_features, train_targets = dataset.prepare_data(
        dataset.train_entries, list(range(len(dataset.train_entries)))
    )
    plot_embedding(train_features, train_targets, args.name + '-before-training')
    print(dataset.hdim, end='\t')
    model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=2 * dataset.hdim)
    model.cuda()
    optimizer = Adam(model.parameters())
    train(model, dataset, optimizer, num_epochs, cuda_device=0, max_patience=5, output_buffer=sys.stderr)
    show_representation(model, dataset.get_next_train_batch, dataset.initialize_train_batches(), 0,
                        args.name + '-after-training-triplet')

    model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=2 * dataset.hdim, lambda1=0, lambda2=0)
    model.cuda()
    optimizer = Adam(model.parameters())
    train(model, dataset, optimizer, num_epochs, cuda_device=0, max_patience=5, output_buffer=sys.stderr)
    show_representation(model, dataset.get_next_train_batch, dataset.initialize_train_batches(), 0,
                        args.name + '-after-training-no-triplet')
    pass
