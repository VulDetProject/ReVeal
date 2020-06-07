import sys

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


class SVMLearningAPI(BaseEstimator):
    def __init__(self, print, balance, model_type='svm'):
        super(SVMLearningAPI, self).__init__()
        self.print = print
        self.balance = balance
        self.model_type = model_type
        if self.print:
            self.output_buffer = sys.stderr
        else:
            self.output_buffer = None
        pass

    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y):
        import warnings
        warnings.filterwarnings('ignore')
        if self.model_type == 'svm':
            self.model = SVC()
        elif self.model_type == 'lr':
            self.model = LogisticRegression()#MLPClassifier(hidden_layer_sizes=(256, 128, 256), max_iter=10)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier()
        if not self.balance:
            full_X, full_Y = train_x, train_y
        else:
            full_X, full_Y = self.rebalance(train_x, train_y)
        if self.output_buffer is not None:
            print('Fitting ' + self.model_type + ' model', file=self.output_buffer)
        self.model.fit(full_X, full_Y)
        if self.output_buffer is not None:
            print('Training Complete', file=self.output_buffer)

    def predict(self, test_x):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        return self.model.predict(test_x)

    def predict_proba(self, test_x):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        return self.model.predict_proba(test_x)

    def evaluate(self, text_x, test_y):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        predictions = self.predict(text_x)
        return {
            'accuracy': accuracy_score(test_y, predictions)*100,
            'precision': precision_score(test_y, predictions)*100,
            'recall': recall_score(test_y, predictions)*100,
            'f1': f1_score(test_y, predictions)*100,
        }


    def score(self, text_x, test_y):
        if not hasattr(self, 'model'):
            raise ValueError('Cannnot call predict or evaluate in untrained model. Train First!')
        scores = self.evaluate(text_x, test_y)
        return scores['f1']

    def rebalance(self, _x, _y):
        smote = SMOTE(random_state=1000)
        return smote.fit_resample(_x, _y)
        pass