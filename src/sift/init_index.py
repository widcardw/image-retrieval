import cv2
import numpy as np
import hnswlib
import os
import time
from tqdm import tqdm
import pandas as pd
from constants import kp_num, dim, scale
import pickle
from sklearn.cluster import KMeans

# https://stackoverflow.com/questions/64525121/sift-surf-module-cv2-cv2-has-no-attribute-xfeatures2d-set-opencv-enabl
sift = cv2.SIFT_create(nfeatures=kp_num)

def _get_image_des(img: cv2.Mat):
    kp, des = sift.detectAndCompute(img, None)
    normalized_array = np.zeros((kp_num, 128))
    if kp is None or des is None:
        return normalized_array
    if des.shape[0] >= kp_num:
        normalized_array = des[:kp_num]
    else:
        normalized_array[:des.shape[0]] = des
    return normalized_array

def img_des_from_path(image_path: str):
    if not os.path.exists(image_path):
        raise Exception(f'{image_path} does not exist!')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return _get_image_des(img)

def img_des_from_bytes(img_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return _get_image_des(img)

def get_all_img_features(dataset_path: str, cluster_num = 100):
    folder_path = os.path.join(dataset_path, 'images')
    all_images = os.listdir(folder_path)
    output_path = os.path.join('output', dataset_path)

    p = os.path.join(output_path, 'sift-des.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as fin:
            descriptors = pickle.load(fin)
    else:
        descriptors = []
        for file in tqdm(all_images):
            full_path = os.path.join(folder_path, file)
            des = img_des_from_path(full_path)
            descriptors.append(des)
        with open(p, 'wb') as fout:
            pickle.dump(descriptors, fout)

    km_pkl = os.path.join(output_path, 'sift-kmeans.pkl')
    if os.path.exists(km_pkl):
        with open(km_pkl, 'rb') as fin:
            kmeans = pickle.load(fin)
    else:
        kmeans = KMeans(n_clusters=cluster_num)
        kmeans.fit(np.concatenate(descriptors))
        with open(os.path.join(output_path, 'sift-kmeans.pkl'), 'wb') as fout:
            pickle.dump(kmeans, fout)

    cluster_centers = kmeans.cluster_centers_

    # each image has a vector of kp_num
    labels = []
    for descriptor in tqdm(descriptors):
        labels.append(kmeans.predict(np.array(descriptor, dtype=np.double)))
    
    with open(os.path.join(output_path, 'sift-kmeans-labels.pkl'), 'wb') as fout:
        pickle.dump(labels, fout)
    with open(os.path.join(output_path, 'sift-kmeans-cluster-centers.pkl'), 'wb') as fout:
        pickle.dump(cluster_centers, fout)
    return labels

def init_index(dataset_folder: str, n_clusters = 100):
    # Image path
    folder_path = os.path.join(dataset_folder, 'images')
    # output path of index file
    output_path = os.path.join('output', dataset_folder)
    if not os.path.exists(folder_path):
        raise Exception(f'Images should be put in {folder_path}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print('Creating dataset')
    all_images = os.listdir(folder_path)

    start_time = time.time()
    pkl = os.path.join(output_path, 'sift-kmeans-labels.pkl')
    # Detect if there is generated pkl file
    # If it exists, just skip the parsing process
    if os.path.exists(pkl):
        with open(pkl, 'rb') as fin:
            data = pickle.load(fin)
        i = len(data)
    # Get the features of each image one by one
    else:
        data = get_all_img_features(dataset_folder, n_clusters)
        i = len(data)

    end_time = time.time()
    print(f"Dataset loaded in: {end_time - start_time}s")

    num_elements = i
    ids = np.arange(num_elements)

    print('Creating index')
    start_time = time.time()
    # Build hnsw index with euclidean distance
    p1 = hnswlib.Index(space='l2', dim=kp_num)
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

    # Save the map from image id to image path
    out_df = pd.DataFrame(enumerate(all_images), columns=['Index', 'File'])
    out_df.to_csv(indexed_image_list_file_path)

