import os

import pandas as pd
import torch

label_path = 'label_names.txt'
label_pd = pd.read_csv(label_path, sep='\t', error_bad_lines=False)

data_path = "empathetic.txt"

data_pd = pd.read_csv(names=['label_id', 'label_name', 'utterance'], filepath_or_buffer=data_path, sep='\t', error_bad_lines=False)

#print(data_pd)

label_ids = torch.randperm(32).tolist()



train_sampled_labels = label_ids[:28]
train_data = data_pd[data_pd['label_id'].isin(train_sampled_labels)]
print(len(train_data))

from sklearn.model_selection import train_test_split

real_train, unseen_test = train_test_split(train_data, test_size=500)

real_train.to_csv('train/train.csv', header=False, index=False, sep='\t')

unseen_test.to_csv('unseen_test/unseen_test.csv', header=False, index=False, sep='\t')

seen_train, seen_test = train_test_split(real_train, test_size=500)

seen_test.to_csv('seen_test/seen_test.csv', header=False, index=False, sep='\t')

print(len(real_train))
print(len(unseen_test))
print(len(seen_test))

test_sampled_labels = label_ids[28:]
test_data = data_pd[data_pd['label_id'].isin(test_sampled_labels)]
for shuffle in range(5):
    for shot in [0, 1, 5]:
        data_frame_list = []
        for label in test_sampled_labels:
            print(label)
            if shot == 0:
                support_train, support_test = train_test_split(test_data[test_data['label_id'] == label],
                                                               test_size=1)
                support_test['utterance'] = support_test['label_name']
            else:
                support_train, support_test = train_test_split(test_data[test_data['label_id']==label], test_size=shot)

            data_frame_list.append(support_test)
        test_pd = pd.concat(data_frame_list)
        if not os.path.exists('support/{0}_{1}/'.format(shuffle,shot)):
            os.mkdir('support/{0}_{1}/'.format(shuffle,shot))
        test_pd.to_csv('support/{0}_{1}/support.csv'.format(shuffle,shot), header=False, index=False, sep='\t')
