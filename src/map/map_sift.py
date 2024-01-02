import numpy as np
import hnswlib
import pandas as pd
import os
import cv2
from tqdm import tqdm

from src.sift.constants import kp_num
from typing import Union, Callable, Tuple, List

import random

sift = cv2.SIFT_create(kp_num)

def calc_sift_map(dataset_name: str, k = 5, calc_ap: Union[Callable, None] = None):
    if calc_ap is None:
        raise Exception('AP calculator not provided!')
    img_map = pd.read_csv(f'output/dataset/{dataset_name}/sift.csv')
    image_path = f'dataset/{dataset_name}/images/'

    p1 = hnswlib.Index(space='l2', dim=kp_num)
    p1.load_index(f'output/dataset/{dataset_name}/sift.bin')
    p1.set_ef(20)

    def query_image(img_path: str) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            return None, None
        feature: np.ndarray = des.flatten()
        labels, distances = p1.knn_query(feature, k)
        return labels.squeeze(), distances.squeeze()
    
    all_images = os.listdir(image_path)
    random.shuffle(all_images)
    all_images = all_images[:200]
    aps: List[float] = []

    for file in tqdm(all_images):
        full_path = os.path.join(image_path, file)
        labels, _ = query_image(full_path)
        if labels is None:
            continue
        search_results = []
        for label in labels:
            search_results.append(img_map.loc[label, 'File'])
        ap = calc_ap(search_results, file)
        aps.append(ap)
    mAP = np.mean(aps)
    with open(f'output/dataset/{dataset_name}/sift-mAP.txt', 'w') as fout:
        fout.write(f'mAP = {mAP}')


