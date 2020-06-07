import copy

import numpy as np
import sys
import torch
from graph_dataset import DataSet
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1
from tqdm import tqdm
from tsne import plot_embedding

from models import MetricLearningModel


def train(model, dataset, optimizer, num_epochs, max_patience=5,
          valid_every=1, cuda_device=-1, output_buffer=sys.stderr):
    if output_buffer is not None:
        print('Start Training', file=output_buffer)
    assert isinstance(model, MetricLearningModel) and isinstance(dataset, DataSet)
    best_f1 = 0
    best_model = None
    patience_counter = 0
    train_losses = []
    try:
        for epoch_count in range(num_epochs):
            batch_losses = []
            num_batches = dataset.initialize_train_batches()
            output_batches_generator = range(num_batches)
            if output_buffer is not None:
                output_batches_generator = tqdm(output_batches_generator)
            for _ in output_batches_generator:
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                features, targets, same_class_features, diff_class_features = dataset.get_next_train_batch()
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                    targets = targets.cuda(device=cuda_device)
                    same_class_features = same_class_features.cuda(device=cuda_device)
                    diff_class_features = diff_class_features.cuda(device=cuda_device)
                probabilities, representation, batch_loss = model(
                    example_batch=features, targets=targets,
                    positive_batch=same_class_features, negative_batch=diff_class_features
                )
                batch_losses.append(batch_loss.detach().cpu().item())
                batch_loss.backward()
                optimizer.step()
            epoch_loss = np.sum(batch_losses).item()
            train_losses.append(epoch_loss)
            if output_buffer is not None:
                print('=' * 100, file=output_buffer)
                print('After epoch %2d Train loss : %10.4f' % (epoch_count, epoch_loss), file=output_buffer)
                print('=' * 100, file=output_buffer)
            if epoch_count % valid_every == 0:
                valid_batch_count = dataset.initialize_valid_batches()
                vacc, vpr, vrc, vf1 = evaluate(
                    model, dataset.get_next_valid_batch, valid_batch_count, cuda_device, output_buffer)
                if vf1 > best_f1:
                    best_f1 = vf1
                    patience_counter = 0
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if dataset.initialize_test_batches() != 0:
                    tacc, tpr, trc, tf1 = evaluate(
                        model, dataset.get_next_test_batch, dataset.initialize_test_batches(), cuda_device,
                        output_buffer=output_buffer
                    )
                    if output_buffer is not None:
                        print('Test Set:       Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % \
                              (tacc, tpr, trc, tf1), file=output_buffer)
                        print('=' * 100, file=output_buffer)
                if output_buffer is not None:
                    print('Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f\tPatience: %2d' % \
                          (vacc, vpr, vrc, vf1, patience_counter), file=output_buffer)
                    print('-' * 100, file=output_buffer)
                if patience_counter == max_patience:
                    if best_model is not None:
                        model.load_state_dict(best_model)
                        if cuda_device != -1:
                            model.cuda(device=cuda_device)
                    break
    except KeyboardInterrupt:
        if output_buffer is not None:
            print('Training Interrupted by User!')
        if best_model is not None:
            model.load_state_dict(best_model)
            if cuda_device != -1:
                model.cuda(device=cuda_device)
    if dataset.initialize_test_batches() != 0:
        tacc, tpr, trc, tf1 = evaluate(
            model, dataset.get_next_test_batch, dataset.initialize_test_batches(), cuda_device)
        if output_buffer is not None:
            print('*' * 100, file=output_buffer)
            print('*' * 100, file=output_buffer)
            print('Test Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f' % \
                  (tacc, tpr, trc, tf1), file=output_buffer)
            print('%f\t%f\t%f\t%f' % (tacc, tpr, trc, tf1))
            print('*' * 100, file=output_buffer)
            print('*' * 100, file=output_buffer)


def predict(model, iterator_function, _batch_count, cuda_device):
    probs = predict_proba(model, iterator_function, _batch_count, cuda_device)
    return np.argmax(probs, axis=-1)


def predict_proba(model, iterator_function, _batch_count, cuda_device):
    model.eval()
    with torch.no_grad():
        predictions = []
        for _ in tqdm(range(_batch_count)):
            features, targets = iterator_function()
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            probs, _, _ = model(example_batch=features)
            predictions.extend(probs)
        model.train()
    return np.array(predictions)


def evaluate(model, iterator_function, _batch_count, cuda_device, output_buffer=sys.stderr):
    if output_buffer is not None:
        print(_batch_count, file=output_buffer)
    model.eval()
    with torch.no_grad():
        predictions = []
        expectations = []
        batch_generator = range(_batch_count)
        if output_buffer is not None:
            batch_generator = tqdm(batch_generator)
        for _ in batch_generator:
            features, targets = iterator_function()
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            probs, _, _ = model(example_batch=features)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()
            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
        model.train()
        return acc(expectations, predictions) * 100, \
               pr(expectations, predictions) * 100, \
               rc(expectations, predictions) * 100, \
               f1(expectations, predictions) * 100,


def show_representation(model, iterator_function, _batch_count, cuda_device, name, output_buffer=sys.stderr):
    model.eval()
    with torch.no_grad():
        representations = []
        expected_targets = []
        batch_generator = range(_batch_count)
        if output_buffer is not None:
            batch_generator = tqdm(batch_generator)
        for _ in batch_generator:
            iterator_values = iterator_function()
            features, targets = iterator_values[0], iterator_values[1]
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            _, repr, _ = model(example_batch=features)
            repr = repr.detach().cpu().numpy()
            print(repr.shape)
            representations.extend(repr.tolist())
            expected_targets.extend(targets.numpy().tolist())
        model.train()
        print(np.array(representations).shape)
        print(np.array(expected_targets).shape)
        plot_embedding(representations, expected_targets, title=name)
