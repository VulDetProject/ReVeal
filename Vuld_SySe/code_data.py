import copy

import sys

import numpy as np
import torch
from gensim.models import Word2Vec


class DataEntry:
    def __init__(self, dataset, sentence, label, meta_data=None, parser=None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet)
        self.sentence = sentence
        self.label = label
        if parser is not None:
            self.words = parser.parse(self.sentence)
        else:
            self.words = self.sentence.split()
        self.meta_data = meta_data
        pass

    def init_word_index(self):
        assert self.dataset.vocab is not None, 'Initialize Dataset Vocabulary First'
        self.word_indices = [self.dataset.vocab.start_token]
        for word in self.words:
            self.word_indices.append(self.dataset.vocab.get_token_id(word))
        self.word_indices.append(self.dataset.vocab.end_token)

    def __repr__(self):
        return str(self.word_indices) + '\t' + str(self.label)

    def __hash__(self):
        return str(self.sentence).__hash__


class DataSet:
    class Vocabulary:
        def __init__(self):
            self.start_token = 0
            self.end_token = 1
            self.pad_token = 2
            self.unk = 3
            self.index_to_token = {0: "<START>", 1:"<END>", 2:"<PAD>", 3: "<UNK>"}
            self.token_to_index = {"<START>": 0, "<END>": 1, "<PAD>": 2, "<UNK>": 3}
            self.count = 4
            pass

        def get_token_id(self, token):
            if token in self.token_to_index.keys():
                return self.token_to_index[token]
            else:
                return self.unk

        def get_token(self, id):
            if id >= self.count:
                return "<UNK>"
            return self.index_to_token[id]

        def add_token(self, token):
            index = self.get_token_id(token)
            if index != self.unk:
                return index
            else:
                index = self.count
                self.count += 1
                self.index_to_token[index] = token
                self.token_to_index[token] = index
                return index

    def __init__(self, initial_embedding_path=None, min_seq_len=-1, intra_dataset=False):
        self.train_entries = []
        self.test_entries = []
        self.dev_entries = []
        self.vocab = None
        self.min_seq_len = min_seq_len
        self.initial_embedding_present = (initial_embedding_path is not None)
        if self.initial_embedding_present:
            self.initial_emddings = Word2Vec.load(initial_embedding_path)
        self.intra_dataset = intra_dataset

    def add_data_entry(self, entry, part='train'):
        assert isinstance(entry, DataEntry)
        if self.initial_embedding_present:
            entry.wvmodel = self.initial_emddings
        if part == 'train':
            if self.intra_dataset:
                if entry in self.train_entries:
                    return False
            self.train_entries.append(entry)
        elif part == 'test':
            self.test_entries.append(entry)
        else:
            self.dev_entries.append(entry)
        return True

    def split_using_fidx(self, p, balance=None):
        print("Splitting based on function", file=sys.stderr)
        fidx_to_entries = {}
        for entry in self.train_entries:
            assert isinstance(entry, DataEntry)
            fidx = entry.meta_data
            if fidx not in fidx_to_entries.keys():
                fidx_to_entries[fidx] = []
            fidx_to_entries[fidx].append(entry)
        findices = list(fidx_to_entries.keys())
        np.random.shuffle(findices)
        test_len = int(len(findices) * p)
        test_findices = findices[:test_len]
        train_findices = findices[test_len:]
        train_entries = []
        self.test_entries = []
        for fidx in train_findices:
            train_entries.extend(fidx_to_entries[fidx])
        for fidx in test_findices:
            self.test_entries.extend(fidx_to_entries[fidx])
        if balance is None:
            self.train_entries = train_entries
        else:
            final_train_entries = []
            positive_entries = []
            negative_entries = []
            for e in train_entries:
                if e.label == 1:
                    positive_entries.append(e)
                else:
                    negative_entries.append(e)
            usample_ratio = balance[0]
            osample_multi = balance[1]
            for e in negative_entries:
                if np.random.uniform(0, 1) <= usample_ratio:
                    final_train_entries.append(e)
            for e in positive_entries:
                for _ in range(osample_multi):
                    final_train_entries.append(e)
            self.train_entries = final_train_entries
        pass

    def split_test_data(self, p, balance=None):
        if self.train_entries[0].meta_data is not None:
            self.split_using_fidx(p, balance=balance)
        else:
            np.random.shuffle(self.train_entries)
            test_len = int(p * len(self.train_entries))
            self.test_entries = self.train_entries[:test_len]
            entries = self.train_entries[test_len:]
            if balance is None:
                self.train_entries = entries
            else:
                final_train_entries = []
                positive_entries = []
                negative_entries = []
                for e in entries:
                    if e.label == 1:
                        positive_entries.append(e)
                    else:
                        negative_entries.append(e)
                usample_ratio = balance[0]
                osample_multi = balance[1]
                for e in negative_entries:
                    if np.random.uniform(0, 1) <= usample_ratio:
                        final_train_entries.append(e)
                for e in positive_entries:
                    for _ in range(osample_multi):
                        final_train_entries.append(e)
                self.train_entries = final_train_entries
        pass

    def get_random_positive_example(self, max_code_len=99999999):
        positive_indices = []
        for eidx, entry in enumerate(self.train_entries):
            if entry.label == 1:
                positive_indices.append(eidx)
        pidx = np.random.choice(positive_indices)
        entry = self.train_entries[pidx]
        while len(entry.word_indices) >= max_code_len:
            pidx = np.random.choice(positive_indices)
            entry = self.train_entries[pidx]
        return entry.word_indices, entry.label, pidx

    def write_examples(self):
        def prepare_code(code):
            if isinstance(code, list):
                code =  ' '.join(code)
            tokens = code.split()
            new_tokens = []
            for t in tokens:
                t = t.strip()
                if t.startswith('F') and t.endswith('('):
                    new_tokens.append(t[:-1])
                else:
                    new_tokens.append(t)
            return ' '.join(new_tokens)

        import os
        from tqdm import tqdm
        d = "for-bert"
        if not os.path.exists(d):
            os.mkdir(d)
        train_file = open(os.path.join(d, 'train.tsv'), 'w')
        test_file = open(os.path.join(d, 'test.tsv'), 'w')
        for eidx, e in enumerate(tqdm(self.train_entries)):
            assert isinstance(e, DataEntry)
            code = prepare_code(e.sentence)
            line = str(eidx) + '\t' + str(e.label) + '\t' + str(e.meta_data) + '\t' + code + '\n'
            train_file.write(line)
        for eidx, e in enumerate(tqdm(self.dev_entries)):
            assert isinstance(e, DataEntry)
            code = prepare_code(e.sentence)
            line = str(eidx) + '\t' + str(e.label) + '\t' + str(e.meta_data) + '\t' + code + '\n'
            train_file.write(line)
        train_file.close()
        for eidx, e in enumerate(tqdm(self.test_entries)):
            assert isinstance(e, DataEntry)
            code = prepare_code(e.sentence)
            line = str(eidx) + '\t' + str(e.label) + '\t' + str(e.meta_data) + '\t' + code + '\n'
            test_file.write(line)
        test_file.close()


    def get_random_negative_example(self, max_code_len=99999999):
        negative_indices = []
        for eidx, entry in enumerate(self.train_entries):
            if entry.label == 0:
                negative_indices.append(eidx)
        nidx = np.random.choice(negative_indices)
        entry = self.train_entries[nidx]
        while len(entry.word_indices) >= max_code_len:
            pidx = np.random.choice(negative_indices)
            entry = self.train_entries[pidx]
        return entry.word_indices, entry.label, nidx

    def get_random_positive_test_example(self, max_code_len=99999999):
        positive_indices = []
        for eidx, entry in enumerate(self.test_entries):
            if entry.label == 1:
                positive_indices.append(eidx)
        pidx = np.random.choice(positive_indices)
        entry = self.test_entries[pidx]
        while len(entry.word_indices) >= max_code_len:
            pidx = np.random.choice(positive_indices)
            entry = self.test_entries[pidx]
        return entry.word_indices, entry.label, pidx

    def get_random_negative_test_example(self,max_code_len=99999999):
        negative_indices = []
        for eidx, entry in enumerate(self.test_entries):
            if entry.label == 0:
                negative_indices.append(eidx)
        nidx = np.random.choice(negative_indices)
        entry = self.test_entries[nidx]
        while len(entry.word_indices) >= max_code_len:
            pidx = np.random.choice(negative_indices)
            entry = self.test_entries[pidx]
        return entry.word_indices, entry.label, nidx

    def convert_word_indices_to_feature_matric(self, word_indices):
        max_seq_len = len(word_indices)
        indices = [self.vocab.pad_token] * max_seq_len
        if self.initial_embedding_present:
            vectors = [np.zeros(shape=self.initial_emddings.vector_size)] * max_seq_len
        for i, w_index in enumerate(word_indices):
            indices[i] = w_index
            if self.initial_embedding_present:
                token = self.vocab.get_token(w_index)
                if token in self.initial_emddings.wv:
                    vectors[i] = self.initial_emddings.wv[token]
                elif '<UNK>' in self.initial_emddings.wv:
                    vectors[i] = self.initial_emddings.wv['<UNK>']
        if self.initial_embedding_present:
            return torch.FloatTensor(np.asarray(vectors))
        else:
            return torch.LongTensor(np.asarray(indices))
        pass

    def init_data_set(self, batch_size=32):
        if len(self.dev_entries) == 0:
            self.dev_entries = self.train_entries[:int(0.1 * len(self.train_entries))]
            self.train_entries = self.train_entries[int(0.1 * len(self.train_entries)):]
        self.build_vocabulary()
        for entry in self.train_entries:
            entry.init_word_index()
        for entry in self.dev_entries:
            entry.init_word_index()
        for entry in self.test_entries:
            entry.init_word_index()
        self.batch_size = batch_size
        self.initialize_batch()

    def build_vocabulary(self):
        self.vocab = DataSet.Vocabulary()
        words = {}
        total_words = 0
        for entry in self.train_entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        for entry in self.test_entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        for entry in self.dev_entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        word_freq = [[key, words[key]] for key in words.keys()]
        word_freq = sorted(word_freq, key=lambda x:x[1], reverse=True)
        accepted_words = word_freq
        for word, count in accepted_words:
            self.vocab.add_token(word)
        # print('Total Number of Words', total_words)
        # print('Unique Words : ', len(words.keys()))
        # print('Vocab Size : ', len(accepted_words))
        # print('=' * 100)
        print('Total Number of Words', total_words, file=sys.stderr)
        print('Unique Words : ', len(words.keys()), file=sys.stderr)
        print('Vocab Size : ', len(accepted_words), file=sys.stderr)
        print('=' * 100, file=sys.stderr)

    def get_data_entries_by_id(self, dataset, ids):
        max_seq_len = max([len(dataset[id].word_indices) for id in ids])
        if self.min_seq_len != -1:
            max_seq_len = max(max_seq_len, self.min_seq_len)
        token_indices = []
        masks = []
        labels = []
        token_vectors = []
        for index in ids:
            indices = [self.vocab.pad_token] * max_seq_len
            if self.initial_embedding_present:
                vectors = [np.zeros(shape=self.initial_emddings.vector_size)] * max_seq_len
            mask = [1] * max_seq_len
            for i, w_index in enumerate(dataset[index].word_indices):
                indices[i] = w_index
                mask[i] = 0
                if self.initial_embedding_present:
                    token = self.vocab.get_token(w_index)
                    if token in self.initial_emddings.wv:
                        vectors[i] = self.initial_emddings.wv[token]
                    elif '<UNK>' in self.initial_emddings.wv:
                        vectors[i] = self.initial_emddings.wv['<UNK>']
            token_indices.append(indices)
            masks.append(mask)
            if self.initial_embedding_present:
                token_vectors.append(vectors)
            labels.append(dataset[index].label)
        if not self.initial_embedding_present:
            return torch.LongTensor(np.asarray(token_indices)), \
                   torch.IntTensor(np.asarray(masks)), \
                   torch.LongTensor(np.asarray(labels))
        else:
            return torch.FloatTensor(np.asarray(token_vectors)), \
                   torch.IntTensor(np.asarray(masks)), \
                   torch.LongTensor(np.asarray(labels))

    def get_train_dataset_by_ids(self, ids):
        return self.get_data_entries_by_id(self.train_entries, ids)

    def get_test_dataset_by_ids(self, ids):
        return self.get_data_entries_by_id(self.test_entries, ids)

    def get_dev_dataset_by_ids(self, ids):
        return self.get_data_entries_by_id(self.dev_entries, ids)

    def initialize_batch(self):
        total = len(self.train_entries)
        indices = np.arange(0,total-1, 1)
        np.random.shuffle(indices)
        self.batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + self.batch_size
            if c_end > end:
                c_end = end
            self.batch_indices.append(indices[curr:c_end])
            curr = c_end

    def get_all_test_examples(self):
        dataset = [None] * len(self.test_entries)
        for i in range(len(self.test_entries)):
            dataset[i] = [self.test_entries[i].meta_data if self.test_entries[i].meta_data is not None else i]
            dataset[i].extend(list(self.get_test_dataset_by_ids([i])))
        return dataset

    def get_all_train_examples(self):
        dataset = [None] * len(self.train_entries)
        for i in range(len(self.train_entries)):
            dataset[i] = [self.train_entries[i].meta_data if self.train_entries[i].meta_data is not None else i]
            dataset[i].extend(list(self.get_train_dataset_by_ids([i])))
        return dataset

    def get_all_dev_examples(self):
        dataset = [None] * len(self.dev_entries)
        for i in range(len(self.dev_entries)):
            dataset[i] = [self.dev_entries[i].meta_data if self.dev_entries[i].meta_data is not None else i]
            dataset[i].extend(list(self.get_dev_dataset_by_ids([i])))
        return dataset

    def get_all_test_batches(self, batch_size=32):
        dataset = []
        indices = [i for i in range(len(self.test_entries))]
        batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            batch_indices.append(indices[curr:c_end])
            curr = c_end
        for indices in batch_indices:
            dataset.append(self.get_test_dataset_by_ids(indices))
        return dataset

    def get_next_batch_train_data(self):
        if len(self.batch_indices) == 0:
            self.initialize_batch()
        indices = self.batch_indices[0]
        self.batch_indices = self.batch_indices[1:]
        return self.get_train_dataset_by_ids(indices)

    def get_batch_count(self):
        return len(self.batch_indices)

    def get_all_batches(self):
        dataset = []
        np.random.shuffle(self.train_entries)
        self.initialize_batch()
        for indices in self.batch_indices:
            dataset.append(self.get_train_dataset_by_ids(indices))
        return dataset

    def get_selective_batches(self, selection=20):
        dataset = []
        self.initialize_batch()
        for idx, indices in enumerate(self.batch_indices):
            dataset.append(self.get_train_dataset_by_ids(indices))
            if idx == selection:
                break
        return dataset

    def get_test_data(self):
        return self.get_data_entries_by_id(self.test_entries, list(range(len(self.test_entries))))

    def get_complete_train_data(self):
        return self.get_data_entries_by_id(self.train_entries, list(range(len(self.train_entries))))

    def get_sentence(self, entries, i):
        return ' '.join(entries[i].words)
        pass

