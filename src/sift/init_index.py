import cv2
import numpy as np
import hnswlib
import os
import time
from tqdm import tqdm
import pandas as pd
from constants import kp_num, dim
import pickle

# https://stackoverflow.com/questions/64525121/sift-surf-module-cv2-cv2-has-no-attribute-xfeatures2d-set-opencv-enabl
sift = cv2.SIFT_create(nfeatures=kp_num)

def _common_get_des(img: cv2.Mat):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    normalized_array = np.zeros((dim,))
    if des is None:
        return normalized_array
    df = des.flatten()
    if len(df) >= dim:
        normalized_array = df[:dim]
    else:
        normalized_array[:len(df)] = df
    return normalized_array

def image_des_from_bytes(img_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return _common_get_des(img)

def image_des(image_path: str):
    if not os.path.exists(image_path):
        raise Exception(f'{image_path} does not exist!')
    img = cv2.imread(image_path)
    return _common_get_des(img)

def init_index(dataset_folder: str):
    folder_path = os.path.join(dataset_folder, 'images')
    output_path = os.path.join('output', dataset_folder)
    if not os.path.exists(folder_path):
        raise Exception(f'Images should be put in {folder_path}')
    
    print('Creating dataset')
    all_images = os.listdir(folder_path)

    start_time = time.time()
    pkl = os.path.join(output_path, 'sift_features.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as fin:
            data = pickle.load(fin)
        i = len(data)
        for j in range(len(data)):
            if len(data[j]) < dim:
                n = np.zeros((dim,))
                n[:len(data[j])] = data[j]
                data[j] = n
    else:
        i = 0
        data = []
        for file in tqdm(all_images):
            des = image_des(os.path.join(folder_path, file))
            data.append(des)
            i = i + 1

        with open(pkl, 'wb') as fout:
            pickle.dump(data, fout)

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

