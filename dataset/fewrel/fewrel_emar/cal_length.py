import pandas as pd


file = "/home/zlii0182/PycharmProjects/Continual_Toolkit/dataset/fewrel/fewrel_emar/train.txt"
cnt = 0
with open(file) as file_in:
    for line in file_in:
        items = line.split('\t')
        if len(items[2].split(' ')) > 48:
            cnt+=1
print(cnt)