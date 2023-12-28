import cv2
import numpy as np
from typing import List
import hnswlib
import os
import time
from tqdm import tqdm
import pandas as pd
from constants import kp_num, dim

# https://stackoverflow.com/questions/64525121/sift-surf-module-cv2-cv2-has-no-attribute-xfeatures2d-set-opencv-enabl
sift = cv2.SIFT_create(nfeatures=kp_num)

def image_des(image_path: str):
    if not os.path.exists(image_path):
        raise Exception(f'{image_path} does not exist!')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return des[:kp_num].flatten()

def init_index(dataset_folder: str):
    folder_path = os.path.join(dataset_folder, 'images')
    output_path = os.path.join('output', dataset_folder)
    if not os.path.exists(folder_path):
        raise Exception(f'Images should be put in {folder_path}')
    
    print('Creating dataset')
    i = 0
    data = []
    start_time = time.time()
    all_images = os.listdir(folder_path)
    for file in tqdm(all_images):
        des = image_des(os.path.join(folder_path, file))
        data.append(des)
        i = i + 1
    end_time = time.time()
    print(f"Dataset created in: {end_time - start_time}s")

    num_elements = i
    ids = np.arange(num_elements)

    print('Creating index')
    start_time = time.time()
    p1 = hnswlib.Index(space='l2', dim=dim)
    print('Init index')
    p1.init_index(max_elements=num_elements, ef_construction=200, M=16)
    print('Inserting data')
    p1.add_items(data, ids)
    end_time = time.time()
    print(f"Index created in: {end_time - start_time}s")
    
    index_file_path = os.path.join(output_path, 'sift.bin')
    indexed_image_list_file_path = os.path.join(output_path, 'sift.csv')
    print('Save index at', index_file_path)
    p1.save_index(index_file_path)

    out_df = pd.DataFrame(enumerate(all_images), columns=['Index', 'File'])
    out_df.to_csv(indexed_image_list_file_path)

