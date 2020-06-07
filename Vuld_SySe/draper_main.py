import json
import sys
import torch
from sklearn import manifold
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import argparse
from code_data import DataSet, DataEntry
from tqdm import tqdm
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from vul_det_models import BiGRUModel, BiLSTMModel, ConvModel, TransformerModel

import test


def train(model, loss_function, optimizer, dataset, num_epochs,
          cuda_device=-1, max_patience=5, dev_score_fn=f1_score):
    model.train()
    softmax = nn.LogSoftmax(dim=-1)
    best_model = None
    best_score = 0
    patience_counter = 0
    for epoch in range(num_epochs):
        try:
            model.train()
            dataset.initialize_batch()
            num_batches = dataset.get_batch_count()
            for _ in tqdm(list(range(num_batches))):
                sequence, _, label = dataset.get_next_batch_train_data()
                mn = torch.min(label).item()
                mx = torch.max(label).item()
                if mn != 0 or mx != 1:
                    continue
                try:
                    model.zero_grad()
                    optimizer.zero_grad()
                    if cuda_device != -1:
                        sequence = sequence.cuda(cuda_device)
                        label = label.cuda(cuda_device)
                    output, features = model(sequence)
                    output = softmax(output)
                    batch_loss = loss_function(output, label)
                    batch_loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    print('=' * 100, file=sys.stderr)
                    print(e, file=sys.stderr)
                    print('Sequence', sequence.size(), file=sys.stderr)
                    print('label', label.size(), file=sys.stderr)
                    print('Output', output.size(), file=sys.stderr)
                    print('=' * 100, file=sys.stderr)
                    sys.stderr.flush()
                    sys.stdout.flush()
            pred, expect = predict(model=model, dataset=dataset, cuda_device=cuda_device, partition='dev')
            score = dev_score_fn(expect, pred)
            if args.test_every_epoch:
                pred, expect = predict(model=model, dataset=dataset, cuda_device=cuda_device, partition='test')
                test_acc, test_f1 = accuracy_score(expect, pred), f1_score(expect, pred)
            else:
                test_acc, test_f1 = 0, 0
            if score > best_score:
                best_score = score
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter == max_patience:
                if best_model is not None:
                    model.load_state_dict(best_model)
                    if cuda_device != -1:
                        model.cuda(cuda_device)
                break
            # print('After Epoch %d\tDev score %0.4f  Patience %d\tTest acc: %0.6f\tF1: %0.6f' %
            #       (epoch + 1, score, patience_counter, test_acc, test_f1))
            # print('#' * 100)
            print('After Epoch %d\tDev score %0.4f  Patience %d\tTest acc: %0.6f\tF1: %0.6f' %
                  (epoch + 1, score, patience_counter, test_acc, test_f1), file=sys.stderr)
            print('#' * 100, file=sys.stderr)
            sys.stdout.flush()
        except KeyboardInterrupt:
            print('Train interrupted by User', file=sys.stderr)
            if best_model is not None:
                model.load_state_dict(best_model)
            break


def generate_embeddings(model, dataset, output_path=None, cuda_device=-1):
    model.eval()
    vectors = []
    with torch.no_grad():
        for sentence, sequence, _, label in tqdm(dataset.get_all_test_examples()):
            if cuda_device != -1:
                sequence = sequence.cuda()
            _, embedding, _ = model(sequence)
            embedding = embedding.cpu().data.numpy().tolist()[0]
            data = {
                'code': sentence,
                'label': label[0].item(),
                'embedding': embedding
            }
            vectors.append(data)
    if output_path is not None:
        json.dump(vectors, open(output_path, 'w'))
    return vectors


def predict(model, dataset, cuda_device=-1, partition='test'):
    model.eval()
    outputs = []
    softmax = nn.LogSoftmax(dim=-1)
    with torch.no_grad():
        if partition == 'test':
            examples = dataset.get_all_test_examples()
        else:
            examples = dataset.get_all_dev_examples()
        for sentence, sequence, _, label in tqdm(examples):
            if cuda_device != -1:
                sequence = sequence.cuda(cuda_device)
            output, features = model(sequence)
            output = softmax(output)
            output = output.cpu().numpy()
            output_labels = np.argmax(output, axis=-1)
            for predicted, expected in zip(output_labels, label):
                outputs.append([predicted, expected])
    outputs = np.asarray(outputs)
    return outputs[:, 0], outputs[:, 1]


def train_rf_model(model, dataset, cuda_device=-1):
    rf_model = RandomForestClassifier(n_estimators=100)
    features = []
    labels = []
    assert isinstance(dataset, DataSet)
    with torch.no_grad():
        model.eval()
        train_examples = dataset.get_all_train_examples()
        for example in train_examples:
            _, sequence, _, label = example
            if cuda_device != -1:
                sequence = sequence.cuda(cuda_device)
            out, example_feature = model(sequence)
            features.append(example_feature.cpu().numpy()[0])
            labels.append(label[0])
    features = np.array(features)
    labels = np.array(labels)
    # plot_embedding(features, labels, 'results/Verum-draper')
    rf_model.fit(features, labels)
    return rf_model
    pass

import matplotlib.pyplot as plt

def plot_embedding(X_org, y, title=None):
    # X, _, Y, _ = train_test_split(X_org, y, test_size=0.7)
    X, Y = X_org, y
    y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    print('Fitting TSNE!', file=sys.stderr)
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(title)
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
    file_ = open(str(title) + '-tsne-results.json', 'w')
    json.dump([X.tolist(), Y.tolist()], file_)
    file_.close()
    print(X.shape, file=sys.stderr)
    for i in range(X.shape[0]):
        if Y[i] == 0:
            plt.text(X[i, 0], X[i, 1], 'o',
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(X[i, 0], X[i, 1], '+',
                     color=plt.cm.Set1(0),
                     fontdict={'weight': 'bold', 'size': 9})
    # plt.scatter()
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title("")
    plt.savefig(title + '.pdf')
    plt.show()

def predict_wirh_rf_model(model, rf_model, dataset, cuda_device, which='vuld'):
    features = []
    expected = []
    assert isinstance(dataset, DataSet)
    # assert isinstance(model, ConvModel)
    with torch.no_grad():
        model.eval()
        train_examples = dataset.get_all_test_examples()
        for example in train_examples:
            _, sequence, _, label = example
            if cuda_device != -1:
                sequence = sequence.cuda(cuda_device)
            out, example_feature = model(sequence)
            features.append(example_feature.cpu().numpy()[0])
            expected.append(label[0])
    features = np.array(features)
    expected = np.array(expected)
    predicted = rf_model.predict(features)
    plot_embedding(features, expected, title=which)
    pairs = [(p, e) for p, e in zip(predicted, expected)]
    scores = []
    count = int(len(pairs) * 0.9)
    for i in range(30):
        np.random.shuffle(pairs)
        taken_pairs = pairs[:count]
        expectations = [p[0] for p in taken_pairs]
        predictions = [p[1] for p in taken_pairs]
        scores.append([accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
            recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100])
    fp = open(which + '.json', 'w')
    all_samples = []
    for ix, (feature, _pred, _expect) in enumerate(zip(features, predicted, expected)):
        all_samples.append({
            'features': feature.tolist(),
            'pred': int(_pred),
            'expect': int(_expect)
        })
        if ix == 0:
            print(all_samples, file=sys.stderr)
    json.dump(all_samples, fp)
    fp.close()
    return predicted, expected, scores
    pass

import scipy.stats as stats

def main(args):
    dataset = DataSet(min_seq_len=10, intra_dataset=args.intra_dataset)
    unknown = 0
    for train_file in args.train_file:
        print('Loading %s train file' % train_file, file=sys.stderr)
        train_data = json.load(open(train_file))
        print('Total Count: %d' % len(train_data), file=sys.stderr)
        for e in train_data:
            if 'tokenized' not in e.keys() or e['tokenized'] is None:
                unknown += 1
                continue
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            if len(e['tokenized'].split()) > 500:
                continue
            entry = DataEntry(dataset, e['tokenized'], label)
            dataset.add_data_entry(entry, part='train')
    if args.dev_file is not None:
        dev_data = json.load(open(args.dev_file))
        for e in dev_data:
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            entry = DataEntry(dataset, e['tokenized'], label)
            dataset.add_data_entry(entry, part='dev')
    if args.test_file is not None:
        test_data = json.load(open(args.test_file))
        for e in test_data:
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            entry = DataEntry(dataset, e['tokenized'], label)
            dataset.add_data_entry(entry, part='test')
    else:
        dataset.split_test_data(p=args.test_percentage, balance=None)
    dataset.init_data_set(batch_size=args.batch_size)
    # print('Train examples : ', len(dataset.train_entries))
    # print('Dev examples : ', len(dataset.dev_entries))
    # print('Test examples : ', len(dataset.test_entries))
    print('Train examples : ', len(dataset.train_entries), file=sys.stderr)
    print('Dev examples : ', len(dataset.dev_entries), file=sys.stderr)
    print('Test examples : ', len(dataset.test_entries), file=sys.stderr)
    emb_dim = args.emb_dim
    model = ConvModel(vocab_size=dataset.vocab.count, emb_dim=emb_dim, pad_idx=2)
    # model = TransformerModel(vocab_size=dataset.vocab.count, emb_dim=emb_dim, pad_idx=2)
    loss_function = nn.NLLLoss()
    optimizer = Adam(model.parameters())
    if args.cuda_device != -1:
        model.cuda(device=args.cuda_device)
    if not args.test_only:
        train(
            model=model, loss_function=loss_function,
            optimizer=optimizer, dataset=dataset,
            num_epochs=args.num_epochs, cuda_device=args.cuda_device
        )
        model_file = open(args.model_path, 'wb')
        torch.save(model, model_file)
        model_file.close()
    else:
        model = torch.load(open(args.model_path, 'rb'))
        if args.cuda_device != -1:
            model.cuda(device=args.cuda_device)
    rf_model = train_rf_model(model, dataset, cuda_device=args.cuda_device)
    if 'devign' in args.model_path:
        exp_name = 'results/draper-devign'
    else:
        exp_name = 'results/draper-verum'
    predictions, expectations, scores = predict_wirh_rf_model(
        model=model, rf_model=rf_model, dataset=dataset, cuda_device=args.cuda_device, which=exp_name)
    scores = np.array(scores)
    all_accuracies = scores[:, 0]
    all_precisions = scores[:, 1]
    all_reacalls = scores[:, 2]
    all_f1s = scores[:, 3]
    # print('*' * 100)
    # print('*' * 100)
    print('%.2f\t%0.2f\t%.2f\t%0.2f\t%.2f\t%0.2f\t%.2f\t%0.2f' \
          % (
              float(np.mean(all_accuracies)), float(np.std(all_accuracies)),
              float(np.mean(all_precisions)), float(np.std(all_precisions)),
              float(np.mean(all_reacalls)), float(np.std(all_reacalls)),
              float(np.mean(all_f1s)), float(np.std(all_f1s))
          ),
          file=sys.stderr
    )
    # print('*' * 100)
    print('%.2f\t%0.2f\t%.2f\t%0.2f\t%.2f\t%0.2f\t%.2f\t%0.2f' \
          % (
              float(np.median(all_accuracies)),
              float(np.quantile(all_accuracies, 0.75) - np.quantile(all_accuracies, 0.25)),
              float(np.median(all_precisions)),
              float(np.quantile(all_precisions, 0.75) - np.quantile(all_precisions, 0.25)),
              float(np.median(all_reacalls)),
              float(np.quantile(all_reacalls, 0.75) - np.quantile(all_reacalls, 0.25)),
              float(np.median(all_f1s)), float(np.quantile(all_f1s, 0.75) - np.quantile(all_f1s, 0.25))
          ),
          file=sys.stderr
    )
    # print('*' * 100)
    # print('*' * 100)
    # print('$'*100)
    # print('='*100)
    print("Test Accuracy: %.3f\tPrecision: %.3f\tRecall: %.3f\tF1 score: %.3f" % (
        accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
        recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100)
    )
    # print('$' * 100)
    # print('=' * 100)
    print('=' * 100, file=sys.stderr)
    print("Test Accuracy: %.3f\tPrecision: %.3f\tRecall: %.3f\tF1 score: %.3f" % (
        accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
        recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100), file=sys.stderr
    )
    print('=' * 100, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',
                        help='Path of the train json file (required).',
                        type=str, default=['../data/draper/train_sampled.json'], nargs='+', required=True)
    parser.add_argument('--dev_file',
                        help='Path of the dev json file.', type=str, default=None)
    parser.add_argument('--test_file',
                        help='Path of the train json file (required, if job=\'generate\' or \'train_and_generate\').',
                        type=str, default=None)
    parser.add_argument('--emb_dim', help='Embedding Dimension', default=13)
    parser.add_argument('--num_epochs', help='Number of Epochs for training.', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size for training.', default=128, type=int)
    parser.add_argument('--model_path', help='Path of the trained model (required).', type=str, default='../models/draper/intra_draper.bin', required=True)
    parser.add_argument('--cuda_device', help='-1 if CPU is used otherwise specific device number.',
                        type=int, default=0)
    parser.add_argument('--test_percentage', default=0.2, type=float)
    parser.add_argument('--test_every_epoch', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--intra_dataset', action='store_true')
    args = parser.parse_args()
    # print(args)
    # print('=' * 100)
    main(args)
