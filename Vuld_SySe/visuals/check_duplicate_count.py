import argparse
import json

def calculate_duplicate(arg):
    examples = []
    for data_file in arg.datasets:
        data = json.load(open(data_file))
        for ex in data:
            if ex['tokenized'] is not None:
                examples.append(ex['tokenized'])
    total_len = len(examples)
    uniq_examples = set(examples)
    uniq_len = len(uniq_examples)
    # print('=' * 100)
    print("%.4f" % (float(total_len - uniq_len) / total_len*100))
    pass

def check_pretrain_dumplicate():
    datasets = [
        # ['../data/SySeVR/API_function_call-processed.json',
        # ['../data/SySeVR/API_function_call-chrome_debian.json',
        # '../data/SySeVR/API_function_call-devign.json']],
        # ['../data/SySeVR/Arithmetic_expression-processed.json',
        # ['../data/SySeVR/Arithmetic_expression-chrome_debian.json',
        # '../data/SySeVR/Arithmetic_expression-devign.json']],
        # ['../data/SySeVR/Array_usage-processed.json',
        # ['../data/SySeVR/Array_usage-chrome_debian.json',
        # '../data/SySeVR/Array_usage-devign.json']],
        ['../data/draper/train_full.json',
        ['../data/draper/valid.json',
        '../data/draper/test.json']]
    ]
    for train_set, test_sets in datasets:
        train_data = json.load(open(train_set))
        train_examples = set()
        all_examples = []
        for cix, ex in enumerate(train_data):
            if cix % 10000 == 0:
                print('train', cix)
            if ex['tokenized'] is not None:
                train_examples.add(ex['tokenized'])
                all_examples.append(ex['tokenized'])
        chrome, devign = test_sets
        chrome_duplicate = 0
        devign_duplicate = 0
        chrome_data = json.load(open(chrome))
        for cix, ex in enumerate(chrome_data):
            if cix % 10000 == 0:
                print('valid', cix)
            if ex['tokenized'] is not None:
                all_examples.append(ex['tokenized'])
                if ex['tokenized'] in train_examples:
                    chrome_duplicate += 1
        devign_data = json.load(open(devign))
        for cix, ex in enumerate(devign_data):
            if cix % 10000 == 0:
                print('test', cix)
            if ex['tokenized'] is not None:
                all_examples.append(ex['tokenized'])
                if ex['tokenized'] in train_examples:
                    devign_duplicate += 1
        chrome_p = float(chrome_duplicate) / len(chrome_data) * 100
        devign_p = float(devign_duplicate) / len(devign_data) * 100
        print(
            train_set, len(train_data), len(train_examples),
            chrome_duplicate, len(chrome_data), devign_duplicate, len(devign_data))
        print(len(all_examples), len(set(all_examples)))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', help='A list of all datasets json files', nargs='+',
        default=[
            # '../../data/VulDeePecker/CWE-119-processed.json',
            # '../../data/VulDeePecker/CWE-399-processed.json',
            # '../../data/SySeVR/API_function_call-processed.json',
            # '../../data/VulDeePecker/chrome_debian.json',
            # '../../data/VulDeePecker/devign.json',
            # '../../data/SySeVR/API_function_call-chrome_debian.json',
            # '../../data/SySeVR/API_function_call-devign.json',
            # '../../data/SySeVR/Arithmetic_expression-processed.json',
            # '../../data/SySeVR/Arithmetic_expression-chrome_debian.json',
            # '../../data/SySeVR/Arithmetic_expression-devign.json',
            # '../../data/SySeVR/Array_usage-processed.json',
            # '../../data/SySeVR/Array_usage-chrome_debian.json',
            # '../../data/SySeVR/Array_usage-devign.json',
            # '../../data/SySeVR/Pointer_usage-processed.json',
            # '../../data/SySeVR/Pointer_usage-chrome_debian.json',
            # '../../data/SySeVR/Pointer_usage-devign.json',
            # '../../data/draper/chrome_debian.json',
            # '../../data/draper/devign.json',
            '../../data/draper/train_full.json',
        ]
    )
    arg = parser.parse_args()
    calculate_duplicate(arg)
    # check_pretrain_dumplicate()
    pass