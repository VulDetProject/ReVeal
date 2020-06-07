import numpy as np
import copy
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import random
import torch
import argparse
import sys
import json
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable

from code_data import DataSet, DataEntry


def initialize_dataset(args):
    inital_emb_path = args.word_to_vec
    dataset = DataSet(initial_embedding_path=inital_emb_path, intra_dataset=False)
    for train_file in args.train_file:
        print('Reading Train File %s' % train_file, file=sys.stderr)
        train_data = json.load(open(train_file))
        print('Total Count: %d' % len(train_data), file=sys.stderr)
        for gidx, e in enumerate(train_data):
            if 'tokenized' not in e.keys() or e['tokenized'] is None:
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
            original_code = e['code']
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=original_code)
            dataset.add_data_entry(entry, part='train')
    if args.dev_file is not None:
        dev_data = json.load(open(args.dev_data))
        for e in dev_data:
            if 'label' in e.keys():
                label = e['label']
            else:
                label = e['leble']
            if not isinstance(label, int):
                continue
            if label > 1:
                label = 1
            original_code = e['code']
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=original_code)
            dataset.add_data_entry(entry, part='dev')
    if args.test_file is not None:
        print('Reading Test File', file=sys.stderr)
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
            original_code = e['code']
            entry = DataEntry(dataset, e['tokenized'], label, meta_data=original_code)
            dataset.add_data_entry(entry, part='test')
    else:
        dataset.split_test_data(p=args.test_percentage)
    dataset.init_data_set(batch_size=1)
    print('Train examples : ', len(dataset.train_entries))
    print('Dev examples : ', len(dataset.dev_entries))
    print('Test examples : ', len(dataset.test_entries))
    print('Train examples : ', len(dataset.train_entries), file=sys.stderr)
    print('Dev examples : ', len(dataset.dev_entries), file=sys.stderr)
    print('Test examples : ', len(dataset.test_entries), file=sys.stderr)
    return dataset


def main(args, dataset):
    softmax = nn.Softmax(dim=-1)
    r = robjects.r
    rpy2.robjects.numpy2ri.activate()
    importr('genlasso')
    importr('gsubfn')
    random.seed(178)
    torch.manual_seed(178)
    model_path = args.model_path
    model = torch.load(open(model_path, 'rb'))
    model.eval()
    torch.no_grad()
    word_indices, actual_label, examples_idx = dataset.get_random_positive_example(100)
    input_series = copy.deepcopy(word_indices)
    input_txt = dataset.convert_word_indices_to_feature_matric(input_series).unsqueeze(0).cuda()
    model_output = model(input_txt)
    if isinstance(model_output, tuple) or isinstance(model_output, list):
        model_output = model_output[0]
    output = softmax(model_output).cpu().detach().numpy()[0]
    prediction_class = np.argmax(output)
    print('Original Prediction class:', prediction_class)
    syth_dat, syth_mat = synthesize(input_series, 5000)
    syth_dat.append(input_series)
    syth_mat.append(np.ones(len(input_series)))
    pred_lst = []
    for dat in tqdm(syth_dat):
        dat = list(dat)
        if len(dat) < 9:
            dat.extend([dataset.vocab.pad_token] * (9 - len(dat)))
        input_txt = dataset.convert_word_indices_to_feature_matric(dat).unsqueeze(0).cuda()
        model_output = model(input_txt)
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            model_output = model_output[0]
        output = softmax(model_output).cpu().detach().numpy()[0]
        prediction_class = np.argmax(output)
        pred_lst.append(prediction_class)
    syth_dat = np.asmatrix(syth_dat)
    syth_mat = np.asarray(syth_mat)
    pred_lst = np.asarray(pred_lst)
    print('Total Positives : ', np.sum(pred_lst))
    X = r.matrix(syth_mat, nrow=syth_mat.shape[0], ncol=syth_mat.shape[1])
    Y = r.matrix(pred_lst, ncol=1)
    r.write(X, 'x.csv')
    n = r.nrow(X)
    p = r.ncol(X)
    results = r.fusedlasso1d(y=Y, X=X, gamma=1, approx=True, maxsteps=2000,
                             minlam=0, rtol=1e-07, btol=1e-07, eps=1e-4)
    result = np.array(r.coef(results, np.sqrt(n * np.log(p)))[0])[:, -1]
    abstract_tokens = []
    concrete_tokens = []
    contributions = []

    sentence = dataset.train_entries[examples_idx].meta_data
    words = ['<START>']
    words.extend(dataset.train_entries[examples_idx].sentence.split())
    words.append('<END>')
    for idx, word, value in zip(word_indices, words, result):
        print(idx, dataset.vocab.get_token(idx), word, value)
        abstract_tokens.append(dataset.vocab.index_to_token[idx])
        concrete_tokens.append(word)
        contributions.append(value)

    # print(len(word_indices), len(words), len(result))
    print('=' * 100)
    print(sentence)
    print(' '.join(abstract_tokens))
    print('=' * 100)
    df = pd.DataFrame({'tok': abstract_tokens, 'value': contributions})
    df.to_csv('./draper.csv', index=False)


def synthesize(input_series, num):
    syth_lst = []
    syth_mat = []
    sample = np.random.randint(1, len(input_series) - 2, num)
    features_range = range(1, len(input_series) - 1)
    for size in sample:
        shut_down = np.random.choice(features_range, size, replace=False)
        tmp = np.asarray(copy.deepcopy(input_series))
        tmp_idx = np.ones(len(input_series))
        tmp = np.delete(tmp, shut_down)
        tmp_idx[shut_down] = 0
        tmp_idx[0] = 0
        tmp_idx[-1] = 0
        syth_lst.append(tmp)
        syth_mat.append(tmp_idx)
    return syth_lst, syth_mat


if __name__ == '__main__':
    test_path = '../../data/draper/devign.json'
    model_path = '../../models/draper_test_devign.bin'
    train_path = '../../data/draper/train_sampled.json'
    # word_to_vec = '../../data/Word2Vec/li_et_al_wv'
    word_to_vec = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_to_vec',
                        help='Path of the Word to vector model.', default=word_to_vec, type=str)
    parser.add_argument('--train_file',
                        help='Path of the train json file (required).',
                        type=str, default=[train_path], nargs='+')
    parser.add_argument('--dev_file',
                        help='Path of the dev json file.', type=str, default=None)
    parser.add_argument('--test_file',
                        help='Path of the train json file (required, if job=\'generate\' or \'train_and_generate\').',
                        type=str, default=test_path)
    parser.add_argument('--model_path', help='Path of the trained model (required).', type=str, default=model_path)
    args = parser.parse_args()
    dataset = initialize_dataset(args)
    # a, b = synthesize([23, 82, 72, 46, 23, 46, 20], 10)
    # print(a)
    # print(b)
    main(args, dataset)
    pass
