import os
import json

directories = ["train", "test"]
for i in directories:
    filenames = os.listdir(i)
    G = dict()
    for filename in filenames:
        if filename.split(".")[1] == 'jpg' or filename.split(".")[1] == 'jpeg' or filename.split(".")[1] == 'JPG' and 'no_pharyngitis' in filename:
            label = 'no_pharyngitis'
            G[filename] = label
        elif 'pharyngitis' in filename:
            label = 'pharyngitis'
            G[filename] = label

    json.dump(G, open(i + ".json", 'w'))

dir = "train"

filenames = os.listdir(dir)

L = list()

for i in filenames:
    if i.split(".")[1] == 'jpg' or i.split(".")[1] == 'jpeg' or i.split(".")[1] == 'JPG':
        L.append(i)

with open("train/names.txt", 'w') as f:
    f.write('\n'.join(L))

'''
import os
import json

dir = "train"
filenames = os.listdir(dir)

L = list()

for i in filenames:
    if(i.split(".")[1] == 'jpg'):
        L.append(i)

with open("names.txt",'w') as f:
    f.write('\n'.join(L))

K = []
counts = {'car':0, 'elephant':0, 'airplane':0, 'dog':0, 'cat':0}
for i in L:
    label = i.split('_')[0]
    if counts[label]!=25:
        K.append(i)
        counts[label] += 1

G = dict()

for i in K:
    label = i.split('_')[0] 
    G[i] = label
    
json.dump( G, open( dir+".json", 'w' ) )
'''
