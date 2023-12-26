import os
import pandas as pd

gt_path = './dataset/oxbuild/gt'

gt_files = os.listdir(gt_path)
columns=['name', 'label']

df = pd.DataFrame(columns=columns)

for file_name in gt_files:
    full_path = os.path.join(gt_path, file_name)
    label = 0
    if file_name.find('good') != -1:
        label = 2
    elif file_name.find('junk') != -1:
        label = 0
    elif file_name.find('ok') != -1:
        label = 1
    else:
        continue
    with open(full_path, 'r') as fin:
        data = list(map(lambda x: [x.strip(), label], fin.readlines()))
    dfn = pd.DataFrame(data, columns=columns)
    df = pd.concat([df, dfn], axis=0)

df = df.drop_duplicates()

df.to_csv('./dataset/oxbuild/gt.csv')
