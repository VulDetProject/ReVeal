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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from vul_det_models import BiGRUModel, BiLSTMModel, ConvModel, TransformerBiGRUModel, TransformerAttentionModel, TransformerPoolModel, BiRNNModel

import test
def plot_embedding(X_org, y, title=None):
    # X, _, Y, _ = train_test_split(X_org, y, test_size=0.7)
    X, Y = X_org[:10000], y[:10000]
    y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    print('Fitting TSNE!', file=sys.stderr)
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(title)
    file_ = open(str(title) + '-tsne-results.json', 'w')
    if isinstance(Y, np.ndarray):
        _y = Y.tolist()
    else:
        _y = Y
    json.dump([X.tolist(), _y], file_)
    file_.close()
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
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
            all_losses = []
            dataset.initialize_batch()
            num_batches = dataset.get_batch_count()
            for _ in tqdm(list(range(num_batches))):
                sequence, _, label = dataset.get_next_batch_train_data()
                # print(torch.min(label), torch.max(label), file=sys.stderr)
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
                    output, _ = model(sequence)
                    output = softmax(output)
                    batch_loss = loss_function(output, label)
                    batch_loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    print('=' * 100, file=sys.stderr)
                    print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
                    print('=' * 100, file=sys.stderr)
                    print('sequence', sequence.shape, file=sys.stderr)
                    print('label', label.shape, file=sys.stderr)
                    print('=' * 100, file=sys.stderr)
                    sys.stderr.flush()
                    sys.stdout.flush()
            pred, expect, _ = predict(model=model, dataset=dataset, cuda_device=cuda_device, partition='dev')
            score = dev_score_fn(expect, pred)
            if args.test_every_epoch:
                pred, expect, _ = predict(model=model, dataset=dataset, cuda_device=cuda_device, partition='test')
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
            print('After Epoch %d\tDev score %0.4f  Patience %d\tTest acc: %0.6f\tF1: %0.6f' %
                  (epoch + 1, score, patience_counter, test_acc, test_f1), file=sys.stderr)
            print('#' * 100, file=sys.stderr)
            sys.stdout.flush()
        except KeyboardInterrupt:
            print('Train interrupetd by User', file=sys.stderr)
            if best_model is not None:
                model.load_state_dict(best_model)
                if cuda_device != -1:
                    model.cuda(cuda_device)
            break

def plot_train_embeddings(model, dataset, cuda_device=-1, which='vuld'):
    model.eval()
    with torch.no_grad():
        dataset.initialize_batch()
        num_batches = dataset.get_batch_count()
        _features = []
        _labels = []
        for _ in tqdm(list(range(num_batches))):
            sequence, _, label = dataset.get_next_batch_train_data()
            if cuda_device != -1:
                sequence = sequence.cuda(cuda_device)
            _, features = model(sequence)
            _features.extend(features.cpu().numpy().tolist())
            _labels.extend(label.cpu().numpy().tolist())
        np.save(which + '-train-features.npy', [_features, _labels])
        plot_embedding(_features, _labels, which)
    pass

def predict(model, dataset, cuda_device=-1, partition='test', which='vuld'):
    model.eval()
    outputs = []
    all_features = []
    all_functions = dict()
    softmax = nn.LogSoftmax(dim=-1)
    with torch.no_grad():
        if partition == 'test':
            examples = dataset.get_all_test_examples()
        else:
            examples = dataset.get_all_dev_examples()
        for fidx, sequence, _, label in tqdm(examples):
            if cuda_device != -1:
                sequence = sequence.cuda(cuda_device)
            output, features = model(sequence)
            if partition == 'test':
                all_features.append(features.cpu().numpy()[0].tolist())
            output = softmax(output)
            output = output.cpu().numpy()
            output_labels = np.argmax(output, axis=-1)
            for predicted, expected in zip(output_labels, label):
                outputs.append([predicted, expected])
            pred = output_labels[0]
            expect = label.item()
            if fidx not in all_functions.keys():
                all_functions[fidx] = {
                    'fidx': fidx,
                    'pred': [],
                    'expect': []
                }
            all_functions[fidx]['pred'].append(pred)
            all_functions[fidx]['expect'].append(expect)

    if partition == 'test':
        results = []
        labels = []
        for ix, (features, label) in enumerate(zip(all_features, outputs)):
            results.append({
                'features': features,
                'pred': int(label[0]),
                'expect': int(label[1])
            })
            labels.append(int(label[1]))
            if ix == 0:
                print(results, file=sys.stderr)
        fp = open(which + '-features.json', 'w')
        json.dump(results, fp)
        fp.close()
        plot_embedding(all_features, labels, title=which)

    outputs = np.asarray(outputs)
    return outputs[:, 0], outputs[:, 1], all_functions



def calculate_function_scores(all_functions, which='vuld'):
    assert isinstance(all_functions, dict)
    predicted_at_least_one_pos = []
    predicted_majority_voring = []
    expected = []
    fp = open(which + '-results.csv', 'w')
    for f in all_functions.keys():
        prediction = all_functions[f]['pred']
        expectations = all_functions[f]['expect'][0]
        expected.append(expectations)
        if max(prediction) == 1:
            fp.write(str(f) + ',1\n')
            predicted_at_least_one_pos.append(1)
        else:
            fp.write(str(f) + ',0\n')
            predicted_at_least_one_pos.append(0)
        if np.sum(prediction) >= float(len(prediction)) / 2.0:
            predicted_majority_voring.append(1)
        else:
            predicted_majority_voring.append(0)
    fp.close()
    pairs_one = [(p, e) for p, e in zip(predicted_at_least_one_pos, expected)]
    scores_one = []
    count = int(len(pairs_one) * 0.9)
    for i in range(30):
        np.random.shuffle(pairs_one)
        taken_pairs = pairs_one[:count]
        expectations = [p[0] for p in taken_pairs]
        predictions = [p[1] for p in taken_pairs]
        scores_one.append(
            [accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
             recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100])

    pairs_majority = [(p, e) for p, e in zip(predicted_majority_voring, expected)]
    scores_mojority = []
    count = int(len(pairs_majority) * 0.9)
    for i in range(30):
        np.random.shuffle(pairs_majority)
        taken_pairs = pairs_one[:count]
        expectations = [p[0] for p in taken_pairs]
        predictions = [p[1] for p in taken_pairs]
        scores_mojority.append(
            [accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
             recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100])
    return accuracy_score(expected, predicted_at_least_one_pos), precision_score(expected, predicted_at_least_one_pos), \
           recall_score(expected, predicted_at_least_one_pos), f1_score(expected, predicted_at_least_one_pos), \
           accuracy_score(expected, predicted_majority_voring), precision_score(expected, predicted_majority_voring), \
            recall_score(expected, predicted_majority_voring), f1_score(expected, predicted_majority_voring),\
            scores_one, scores_mojority
    pass


def main(args):
    inital_emb_path = args.word_to_vec
    dataset = DataSet(initial_embedding_path=inital_emb_path, intra_dataset=args.intra_dataset)
    findices = set()
    nindices = set()
    v_nt = 0
    nv_nt = 0
    for train_file in args.train_file:
        print('Reading Train File %s' % train_file, file=sys.stderr)
        train_data = json.load(open(train_file))
        print('Total Count: %d' % len(train_data), file=sys.stderr)
        for e in tqdm(train_data):
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            fidx = e['fidx'] if 'fidx' in e.keys() else None
            if len(e['tokenized'].split()) > 500:
                if fidx not in nindices:
                    if label == 1:
                        v_nt += 1
                    else:
                        nv_nt += 1
                nindices.add(fidx)
                continue
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=fidx)
            added = dataset.add_data_entry(entry, part='train')
            if added and fidx is not None:
                findices.add(fidx)
    if args.dev_file is not None:
        dev_data = json.load(open(args.dev_data))
        for e in tqdm(dev_data):
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            fidx = e['fidx'] if 'fidx' in e.keys() else None
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=fidx)
            dataset.add_data_entry(entry, part='dev')
    if args.test_file is not None:
        print('Reading Test File', file=sys.stderr)
        test_data = json.load(open(args.test_file))
        for e in tqdm(test_data):
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            fidx = e['fidx'] if 'fidx' in e.keys() else None
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=fidx)
            dataset.add_data_entry(entry, part='test')
    else:
        dataset.split_test_data(p=args.test_percentage)
    dataset.init_data_set(batch_size=args.batch_size)
    # print('Train examples : ', len(dataset.train_entries))
    # print('Dev examples : ', len(dataset.dev_entries))
    # print('Test examples : ', len(dataset.test_entries))
    # print('Total unique examples :', len(findices))
    print('Train examples : ', len(dataset.train_entries), file=sys.stderr)
    print('Dev examples : ', len(dataset.dev_entries), file=sys.stderr)
    print('Test examples : ', len(dataset.test_entries), file=sys.stderr)
    print('Total unique examples :', len(findices), file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    emb_dim = dataset.initial_emddings.vector_size
    dataset.write_examples()
    if args.model_type == 'bigru':
        model = BiGRUModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    elif args.model_type == 'bilstm':
        model = BiLSTMModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    elif args.model_type == 'transgru':
        model = TransformerBiGRUModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    elif args.model_type == 'transattn':
        model = TransformerAttentionModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    elif args.model_type == 'transpool':
        model = TransformerPoolModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    elif args.model_type == 'birnn':
        model = BiRNNModel(emb_dim=emb_dim, hidden_size=args.hidden_size, num_layer=args.num_layers)
    else:
        raise ValueError('Invalid Model')
    print(model, file=sys.stderr)
    loss_function = nn.NLLLoss()
    optimizer = Adam(model.parameters())
    if args.cuda_device != -1:
        model.cuda(device=args.cuda_device)
    if not args.test_only:
        print('Starting Training', file=sys.stderr)
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

    if 'SySe' in args.model_path:
        if 'devign' in args.model_path:
            exp_name = 'results/Devign-SySeVR'
        else:
            exp_name = 'results/Verum-SySeVR'
    else:
        if 'devign' in args.model_path:
            exp_name = 'results/Devign-VD'
        else:
            exp_name = 'results/Verum-VD'
    predictions, expectations, all_functions = predict(
        model=model, dataset=dataset, cuda_device=args.cuda_device, which=exp_name)
    # plot_train_embeddings(model=model, dataset=dataset, cuda_device=args.cuda_device, which=exp_name)
    facc_1, fpr_1, frc_1, ff1_1, facc_m, fpr_m, frc_m, ff1_m, scores, scores_m = calculate_function_scores(all_functions, which=exp_name)
    # print('$'*100)
    # print('='*100)
    print(accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
        recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100,
          facc_1*100, fpr_1*100, frc_1*100, ff1_1*100, facc_m * 100, fpr_m * 100, frc_m * 100, ff1_m * 100, sep='\t')
    # print("Test Accuracy: %.3f\tPrecision: %.3f\tRecall: %.3f\tF1 score: %.3f" %(
    #     accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
    #     recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100)
    # )
    # # print('$' * 100)
    # print("Func Accuracy One: %.3f\tFPrecision: %.3f\tFRecall: %.3f\tFF1 score: %.3f" % (facc_1*100, fpr_1*100, frc_1*100, ff1_1*100))
    # # print('$' * 100)
    # print("Func Accuracy Majority: %.3f\tFPrecision: %.3f\tFRecall: %.3f\tFF1 score: %.3f" % (facc_m * 100, fpr_m * 100, frc_m * 100, ff1_m * 100))
    # # print('$' * 100)
    # print('=' * 100)
    print("Test Accuracy: %.3f\tPrecision: %.3f\tRecall: %.3f\tF1 score: %.3f" % (
        accuracy_score(expectations, predictions) * 100, precision_score(expectations, predictions) * 100,
        recall_score(expectations, predictions) * 100, f1_score(expectations, predictions) * 100), file=sys.stderr
          )
    print('$' * 100, file=sys.stderr)
    print("Func Accuracy One: %.3f\tFPrecision: %.3f\tFRecall: %.3f\tFF1 score: %.3f" % (
    facc_1 * 100, fpr_1 * 100, frc_1 * 100, ff1_1 * 100), file=sys.stderr)
    print('$' * 100, file=sys.stderr)
    print("Func Accuracy Majority: %.3f\tFPrecision: %.3f\tFRecall: %.3f\tFF1 score: %.3f" % (
    facc_m * 100, fpr_m * 100, frc_m * 100, ff1_m * 100), file=sys.stderr)
    print('$' * 100, file=sys.stderr)

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
    # print('*' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_to_vec',
                        help='Path of the Word to vector model.', default='../data/Word2Vec/li_et_al_wv', type=str, required=True)
    parser.add_argument('--train_file',
                        help='Path of the train json file (required).',
                        type=str, default=['../data/SySeVR/Array_usage-processed.json'], nargs='+', required=True)
    parser.add_argument('--dev_file',
                        help='Path of the dev json file.', type=str, default=None)
    parser.add_argument('--test_file',
                        help='Path of the train json file (required, if job=\'generate\' or \'train_and_generate\').',
                        type=str, default=None)
    parser.add_argument('--model_type', help='Type of the model (bigru, bilstm)', default='bilstm',
                        choices=['bilstm', 'bigru', 'transgru', 'transattn', 'transpool', 'birnn'])
    parser.add_argument('--num_layers', help='Number of layers', default=2)
    parser.add_argument('--hidden_size', help='Model dimension.', default=256, type=int)
    parser.add_argument('--num_epochs', help='Number of Epochs for training.', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size for training.', default=128, type=int)
    parser.add_argument('--model_path', help='Path of the trained model (required).', type=str, required=True)
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
