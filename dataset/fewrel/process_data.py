import numpy as np
import random

# split training data into valid data
data = []
file_name = "./train_with_entity.txt"
with open(file_name) as file_in:
    for line in file_in:
        data.append(line)

print(len(data))

data1 = []
data2 = []
index = list(range(len(data)))
random.shuffle(index)
length = int(len(data) / 4)
index1 = index[:length]
index2 = index[length:]
for i in index1:
    data1.append(data[i])
for i in index2:
    data2.append(data[i])

with open("./valid_with_entity_sampled.txt", "w") as file_out:
    for i in data1:
        file_out.write(i)

with open("./train_with_entity_sampled.txt", "w") as file_out:
    for i in data2:
        file_out.write(i)

data = []
file_name = "../fewrel_emar/train.txt"
with open(file_name) as file_in:
    for line in file_in:
        data.append(line)

print(len(data))

data1 = []
data2 = []

for i in index1:
    data1.append(data[i])
for i in index2:
    data2.append(data[i])

with open("../fewrel_emar/valid_sampled.txt", "w") as file_out:
    for i in data1:
        file_out.write(i)

with open("../fewrel_emar/train_sampled.txt", "w") as file_out:
    for i in data2:
        file_out.write(i)
        

file_name = "./rel_cluster_label.npy"
data = np.arange(10).repeat(8)
np.random.shuffle(data)
print(data)
np.save("../fewrel_emar/rel_cluster_label_random.npy", data)
np.save("../fewrel_cili/rel_cluster_label_random.npy", data)
