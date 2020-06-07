import numpy
import sys
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from graph_dataset import DataSet
from models import MetricLearningModel
from trainer import train, predict, predict_proba, evaluate as evaluate_from_model


class RepresentationLearningModel(BaseEstimator):
    def __init__(self,
                 alpha=0.5, lambda1=0.5, lambda2=0.001, hidden_dim=256,  # Model Parameters
                 dropout=0.2, batch_size=64, balance=True,   # Model Parameters
                 num_epoch=100, max_patience=20,  # Training Parameters
                 print=False, num_layers=1
                 ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dropout = dropout
        self.num_epoch = num_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size
        self.balance = balance
        self.cuda = torch.cuda.is_available()
        self.print = print
        self.num_layers = num_layers
        if print:
            self.output_buffer = sys.stderr
        else:
            self.output_buffer = None
        pass

    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y):
        input_dim = train_x.shape[1]
        self.model = MetricLearningModel(
            input_dim=input_dim, hidden_dim=self.hidden_dim, aplha=self.alpha, lambda1=self.lambda1,
            lambda2=self.lambda2, dropout_p=self.dropout, num_layers=self.num_layers
        )
        self.optimizer = Adam(self.model.parameters())
        if self.cuda:
            self.model.cuda(device=0)
        self.dataset = DataSet(self.batch_size, train_x.shape[1])
        for _x, _y in zip(train_x, train_y):
            if numpy.random.uniform() <= 0.1:
                self.dataset.add_data_entry(_x.tolist(), _y.item(), 'valid')
            else:
                self.dataset.add_data_entry(_x.tolist(), _y.item(), 'train')
        self.dataset.initialize_dataset(balance=self.balance, output_buffer=self.output_buffer)
        train(
            model=self.model, dataset=self.dataset, optimizer=self.optimizer,
            num_epochs=self.num_epoch, max_patience=self.max_patience,
            cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        if self.output_buffer is not None:
            print('Training Complete', file=self.output_buffer)

    def predict(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return predict(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
        )

    def predict_proba(self, text_x):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part='test')
        return predict_proba(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1
        )

    def evaluate(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        acc, pr, rc, f1 = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return {
            'accuracy': acc,
            'precision': pr,
            'recall': rc,
            'f1': f1
        }

    def score(self, text_x, test_y):
        if not hasattr(self, 'dataset'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part='test')
        _, _, _, f1 = evaluate_from_model(
            model=self.model, iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(), cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer
        )
        return f1