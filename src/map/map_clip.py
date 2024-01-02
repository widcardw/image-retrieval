import numpy as np
import hnswlib
from towhee import pipe, ops
import os
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Union, Callable
from src.clip.constants import model_name
import random

p = (
    pipe.input('path')
        .map('path', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_text_embedding.clip(model_name=model_name, modality='image'))
        .map('embedding', 'embedding', lambda x:x/np.linalg.norm(x))  # 归一化操作
        .output('embedding')
)

def calc_clip_map(dataset_name: str, k = 5, calc_ap: Union[Callable, None] = None):
    if calc_ap is None:
        raise Exception('AP calculator not provided!')
    img_map = pd.read_csv(f'output/dataset/{dataset_name}/clip.csv')
    image_path = f'dataset/{dataset_name}/images/'

    dim = 512

    p1 = hnswlib.Index(space = 'l2', dim = dim)
    p1.load_index(f'output/dataset/{dataset_name}/clip.bin')
    p1.set_ef(20)

    def query_image(img_path: str) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        res = p(img_path).get()
        vec = res[0]
        labels, distances = p1.knn_query(vec, k)
        return labels.squeeze(), distances.squeeze()
    
    all_images = os.listdir(image_path)
    random.shuffle(all_images)
    all_images = all_images[:200]
    aps: List[float] = []
    for file in tqdm(all_images):
        full_path = os.path.join(image_path, file)
        labels, _ = query_image(full_path)
        search_results = []
        for label in labels:
            search_results.append(img_map.loc[label, 'File'])
        ap = calc_ap(search_results, file)
        aps.append(ap)
    mAP = np.mean(aps)
    with open(f'output/dataset/{dataset_name}/clip-mAP.txt', 'w') as fout:
        fout.write(f'mAP = {mAP}')

    ap_df = pd.DataFrame(zip(all_images, aps), columns=['image', 'ap'])
    ap_df.to_csv(f'output/dataset/{dataset_name}/clip-AP.csv')
