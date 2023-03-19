import operator

import pandas as pd
import torch


def random_shuffle(data_list, random_index):
    return [data_list[id] for id in random_index]
data_path = "all_train.csv"

data_pd = pd.read_csv(data_path, sep=',', error_bad_lines=False)


example_list = set()
label_dict = {}
for index, row in data_pd.iterrows():
    #print(row[0])
    label_text = row[2]
    utterance = row[3].replace('_comma_', ',')
    if not label_text in label_dict:
        label_dict[label_text] = len(label_dict.keys())
    example_list.add((label_dict[label_text], label_text, utterance))


dataset_number = len(example_list)
random_index = torch.randperm(dataset_number).tolist()

label_id_list = [example[0] for example in example_list]

label_text_list = [example[1] for example in example_list]

utterance_list = [example[2] for example in example_list]

label_id_list = random_shuffle(label_id_list, random_index)

label_text_list = random_shuffle(label_text_list, random_index)

utterance_list = random_shuffle(utterance_list, random_index)

new_pd = pd.DataFrame(data={'label_id':label_id_list, 'label_name':label_text_list, 'utterance':utterance_list})
new_pd.to_csv('empathetic.txt', header=False, index=False, sep='\t')


label_name_list = [key for key, value in sorted(label_dict.items(), key=operator.itemgetter(1))]

label_names = pd.DataFrame(data={'label_name':label_name_list})
label_names.to_csv('label_names.txt', header=False, index=False, sep='\t')


data_length = len(label_id_list)

train_length = int(0.7*data_length)
valid_length = int(0.8*data_length) - train_length

new_pd = pd.DataFrame(data={'label_id':label_id_list[:train_length], 'label_name':label_text_list[:train_length], 'utterance':utterance_list[:train_length]})
new_pd.to_csv('train.csv', header=False, index=False, sep='\t' )


new_pd = pd.DataFrame(data={'label_id':label_id_list[train_length:train_length+valid_length], 'label_name':label_text_list[train_length:train_length+valid_length], 'utterance':utterance_list[train_length:train_length+valid_length]})
new_pd.to_csv('valid.csv', header=False, index=False, sep='\t')


new_pd = pd.DataFrame(data={'label_id':label_id_list[train_length+valid_length:], 'label_name':label_text_list[train_length+valid_length:], 'utterance':utterance_list[train_length+valid_length:]})
new_pd.to_csv('test.csv', header=False, index=False, sep='\t' )