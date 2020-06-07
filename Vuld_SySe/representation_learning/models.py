import numpy as np
import torch
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
from torch import nn
from torch.optim import Adam

from tsne import plot_embedding


class MetricLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.2, aplha=0.5, lambda1=0.5, lambda2=0.001, num_layers=1):
        super(MetricLearningModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.dropout_p = dropout_p
        self.alpha = aplha
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            nn.LogSoftmax(dim=-1)
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_function = nn.NLLLoss(reduction='none')
        # print(self.alpha, self.lambda1, self.lambda2, sep='\t', end='\t')

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(self, example_batch,
                targets=None,
                positive_batch=None,
                negative_batch=None):
        train_mode = (positive_batch is not None and
                      negative_batch is not None and
                      targets is not None)
        h_a = self.extract_feature(example_batch)
        y_a = self.classifier(h_a)
        probs = torch.exp(y_a)
        batch_loss = None
        if targets is not None:
            ce_loss = self.loss_function(input=y_a, target=targets)
            batch_loss = ce_loss.sum(dim=-1)
        if train_mode:
            h_p = self.extract_feature(positive_batch)
            h_n = self.extract_feature(negative_batch)
            dot_p = h_a.unsqueeze(dim=1) \
                .bmm(h_p.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            dot_n = h_a.unsqueeze(dim=1) \
                .bmm(h_n.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            mag_a = torch.norm(h_a, dim=-1)
            mag_p = torch.norm(h_p, dim=-1)
            mag_n = torch.norm(h_n, dim=-1)
            D_plus = 1 - (dot_p / (mag_a * mag_p))
            D_minus = 1 - (dot_n / (mag_a * mag_n))
            trip_loss = self.lambda1 * torch.abs((D_plus - D_minus + self.alpha))
            ce_loss = self.loss_function(input=y_a, target=targets)
            l2_loss = self.lambda2 * (mag_a + mag_p + mag_n)
            total_loss = ce_loss + trip_loss + l2_loss
            batch_loss = (total_loss).sum(dim=-1)
        return probs, h_a, batch_loss
        pass


if __name__ == '__main__':
    np.random.rand(1000)
    torch.manual_seed(1000)
    batch_size = 128
    input_dim = 200
    hdim = 256
    alpha = 0.1
    x_a = torch.randn(size=[batch_size+32, input_dim])
    test_x = x_a[batch_size:, :]
    x_a = x_a[:batch_size, :]
    targets = torch.randint(0, 2, size=[batch_size + 32])
    test_y = targets[batch_size:]
    targets = targets[:batch_size]
    x_p = torch.randn(size=[batch_size, input_dim])
    x_n = torch.randn(size=[batch_size, input_dim])

    model = MetricLearningModel(input_dim=input_dim, hidden_dim=hdim)
    # print(model)
    optimizer = Adam(model.parameters())

    for epoch in range(50):
        model.zero_grad()
        optimizer.zero_grad()
        prediction_prob, representation, batch_loss = model(
            example_batch=x_a,
            targets=targets,
            positive_batch=x_p,
            negative_batch=x_n)
        repr = representation.detach().cpu().numpy()
        prediction_classes = np.argmax(prediction_prob.detach().cpu().numpy(), axis=-1)
        # print(
        #     "Epoch %3d, Loss: %10.4f, Accuracy: %5.2f, Precision: %5.2f, Recall: %5.2f, F1: %5.2f" % (
        #         epoch, batch_loss.detach().cpu().item(),
        #         acc(targets, prediction_classes), pr(targets, prediction_classes),
        #         rc(targets, prediction_classes), f1(targets, prediction_classes)
        #     )
        # )
        if epoch % 1 == 0:
            prediction_prob, representation, batch_loss = model(
                example_batch=test_x,
                targets=test_y)
            repr = representation.detach().cpu().numpy()
            prediction_classes = np.argmax(prediction_prob.detach().cpu().numpy(), axis=-1)
            print('=' * 100)
            print(
                "Test  %3d, Loss: %10.4f, Accuracy: %5.2f, Precision: %5.2f, Recall: %5.2f, F1: %5.2f" % (
                    epoch, batch_loss.detach().cpu().item(),
                    acc(test_y, prediction_classes), pr(test_y, prediction_classes),
                    rc(test_y, prediction_classes), f1(test_y, prediction_classes)
                )
            )
            print('=' * 100)
            plot_embedding(repr, test_y, title='Epoch %d' % epoch)
        batch_loss.backward()
        optimizer.step()
    pass
