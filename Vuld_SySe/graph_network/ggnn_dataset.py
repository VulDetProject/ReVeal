import torch
from typing import List

import sys
import numpy as np


class AdjacencyList:
    """represent the topology of a graph"""
    def __init__(self, node_num: int, adj_list: List, device: torch.device):
        self.node_num = node_num
        self.data = torch.tensor(adj_list, dtype=torch.long, device=device)
        self.edge_num = len(adj_list)

    @property
    def device(self):
        return self.data.device

    def __getitem__(self, item):
        return self.data[item]


class GGNNEntry:
    def __init__(self, dataset, feature_repr, label, meta_data=None):
        self.dataset = dataset
        assert isinstance(self.dataset, GG)
        self.features = copy.deepcopy(feature_repr)
        self.label = label
        self.meta_data = meta_data
        pass

    def __repr__(self):
        return str(self.features) + '\t' + str(self.label)

    def __hash__(self):
        return str(self.features).__hash__

    def is_positive(self):
        return self.label == 1


class GGNNDataset:
    def __init__(self, batch_size, hdim):
        self.train_entries = []
        self.valid_entries = []
        self.test_entries = []
        self.train_batch_indices = []
        self.valid_batch_indices = []
        self.test_batch_indices = []
        self.batch_size = batch_size
        self.hdim = hdim
        self.positive_indices_in_train = []
        self.negative_indices_in_train = []

    def initialize_dataset(self, balance=True, output_buffer=sys.stderr):
        # if isinstance(balance, bool) and balance:
        #     entries = []
        #     train_features = []
        #     train_targets = []
        #     for entry in self.train_entries:
        #         train_features.append(entry.features)
        #         train_targets.append(entry.label)
        #     train_features = np.array(train_features)
        #     train_targets = np.array(train_targets)
        #     smote = SMOTE(random_state=1000)
        #     features, targets = smote.fit_resample(train_features, train_targets)
        #     for feature, target in zip(features, targets):
        #         entries.append(DataEntry(self, feature.tolist(), target.item()))
        #     self.train_entries = entries
        # elif isinstance(balance, list) and len(balance) == 2:
        #     entries = []
        #     for entry in self.train_entries:
        #         if entry.is_positive():
        #             for _ in range(balance[0]):
        #                 entries.append(
        #                     DataEntry(self, entry.features, entry.label, entry.meta_data)
        #                 )
        #         else:
        #             if np.random.uniform() <= balance[1]:
        #                 entries.append(
        #                     DataEntry(self, entry.features, entry.label, entry.meta_data)
        #                 )
        #     self.train_entries = entries
        #     pass
        for tidx, entry in enumerate(self.train_entries):
            if entry.label == 1:
                self.positive_indices_in_train.append(tidx)
            else:
                self.negative_indices_in_train.append(tidx)
        self.initialize_train_batches()
        if output_buffer is not None:
            print('Number of Train Entries %d #Batches %d' % \
                  (len(self.train_entries), len(self.train_batch_indices)), file=output_buffer)
        self.initialize_valid_batches()
        if output_buffer is not None:
            print('Number of Valid Entries %d #Batches %d' % \
                  (len(self.valid_entries), len(self.valid_batch_indices)), file=output_buffer)
        self.initialize_test_batches()
        if output_buffer is not None:
            print('Number of Test  Entries %d #Batches %d' % \
                  (len(self.test_entries), len(self.test_batch_indices)), file=output_buffer)

    def add_data_entry(self, ggnn_graph_entry, part='train'):
        assert part in ['train', 'valid', 'test']
        if part == 'train':
            self.train_entries.append(ggnn_graph_entry)
        elif part == 'valid':
            self.valid_entries.append(ggnn_graph_entry)
        else:
            self.test_entries.append(ggnn_graph_entry)

    def initialize_train_batches(self):
        self.train_batch_indices = self.create_batches(self.batch_size, self.train_entries)
        return len(self.train_batch_indices)
        pass

    def clear_test_set(self):
        self.test_entries = []

    def initialize_valid_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batch_indices = self.create_batches(batch_size, self.valid_entries)
        return len(self.valid_batch_indices)
        pass

    def initialize_test_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batch_indices = self.create_batches(batch_size, self.test_entries)
        return len(self.test_batch_indices)
        pass

    def get_next_train_batch(self):
        if len(self.train_batch_indices) > 0:
            indices = self.train_batch_indices.pop()
            features, targets = self.prepare_data(self.train_entries, indices)
            same_class_features = self.find_same_class_data(ignore_indices=indices)
            different_class_features = self.find_different_class_data(ignore_indices=indices)
            return features, targets, same_class_features, different_class_features
        raise ValueError('Initialize Train Batch First by calling dataset.initialize_train_batches()')
        pass

    def get_next_valid_batch(self):
        if len(self.valid_batch_indices) > 0:
            indices = self.valid_batch_indices.pop()
            return self.prepare_data(self.valid_entries, indices)
        raise ValueError('Initialize Valid Batch First by calling dataset.initialize_valid_batches()')
        pass

    def get_next_test_batch(self):
        if len(self.test_batch_indices) > 0:
            indices = self.test_batch_indices.pop()
            return self.prepare_data(self.test_entries, indices)
        raise ValueError('Initialize Test Batch First by calling dataset.initialize_test_batches()')
        pass

    def create_batches(self, batch_size, entries):
        _batches = []
        if batch_size == -1:
            batch_size = self.batch_size
        total = len(entries)
        indices = np.arange(0, total - 1, 1)
        np.random.shuffle(indices)
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            _batches.append(indices[curr:c_end])
            curr = c_end
        return _batches
