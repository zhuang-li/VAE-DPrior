import csv

import pandas as pd

def transform_pd(data_pd, label_texts):
    example_list = []

    for index, row in data_pd.iterrows():
        # print(row[0])
        label_id = row[0] - 1
        negative_label = row[1]
        utterance = row[2]
        example_list.append((label_id, label_texts[label_id], utterance))

    return example_list

def to_path(examples, path):
    label_id_list = [label_id for label_id, label_text, utterance in examples]
    label_text_list = [label_text for label_id, label_text, utterance in examples]
    utterance_list = [utterance for label_id, label_text, utterance in examples]

    new_pd = pd.DataFrame(data={'label_id': label_id_list, 'label_name': label_text_list, 'utterance': utterance_list})
    new_pd.to_csv(path, header=False, index=False, sep='\t')

label_path = "label_names.txt"

train_path = "train.txt"

test_path = "train.txt"

valid_path = "valid.txt"


label_texts = pd.read_csv(filepath_or_buffer=label_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')[0].tolist()

train_pd = pd.read_csv(filepath_or_buffer=train_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')


test_pd = pd.read_csv(filepath_or_buffer=test_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')

valid_pd = pd.read_csv(filepath_or_buffer=valid_path, sep='\t', header=None, lineterminator='\n',
                             quoting=csv.QUOTE_NONE, encoding='utf-8')

train_examples = transform_pd(train_pd, label_texts)

test_examples = transform_pd(test_pd, label_texts)

valid_examples = transform_pd(valid_pd, label_texts)

train_path = "classifier/train.csv"

test_path = "classifier/test.csv"

valid_path = "classifier/valid.csv"

whole_path = "personality.txt"

to_path(train_examples, train_path)

to_path(test_examples, test_path)

to_path(valid_examples, valid_path)

to_path(train_examples + test_examples + valid_examples, whole_path)
