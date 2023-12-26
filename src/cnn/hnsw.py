import hnswlib
import numpy as np
import pickle
from use_pretrained import get_image_feature
from PIL import Image

dim = 512
with open('output/dataset/oxbuild/image_list.txt', 'r') as fin:
    image_list = fin.readlines()
with open('output/dataset/oxbuild/features.pkl', 'rb') as fin:
    data: np.ndarray = pickle.load(fin)

num_elements = len(image_list)
ids = np.arange(num_elements)

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=16)
p.add_items(data, ids)

target_img = Image.open('dataset/oxbuild/images/all_souls_000134.jpg')
feat = get_image_feature(target_img)
labels, distances = p.knn_query(feat, k=10)

print(labels)

res = []
for i in labels.squeeze():
    res.append(image_list[i])
print(res)
