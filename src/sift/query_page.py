import streamlit as st

import numpy as np
import hnswlib
import pandas as pd
import os
import time
import cv2
import pickle

from src.sift.constants import kp_num
from typing import Union, Callable, List
from sklearn.cluster import KMeans

sift = cv2.SIFT_create(kp_num)

def calc_map(ap: List[float]):
    if len(ap) == 0:
        return 0.
    return sum(ap) / len(ap)

aps = []

def subpage_sift(dataset_name: str, title = '', calc_ap: Union[Callable, None] = None):
    img_map = pd.read_csv(f'output/dataset/{dataset_name}/sift.csv')
    image_path = f'dataset/{dataset_name}/images/'
    with open(f'output/dataset/{dataset_name}/sift-kmeans.pkl', 'rb') as fin:
        kmeans: KMeans = pickle.load(fin)

    p1 = hnswlib.Index(space='l2', dim=kp_num)
    p1.load_index(f'output/dataset/{dataset_name}/sift.bin')
    p1.set_ef(20)

    st.title(title if title.strip() != '' else dataset_name)
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    k = st.slider('Result count', min_value=1, max_value=15, step=1, value=3)

    def query_image(img_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        kp, des = sift.detectAndCompute(img, None)
        normalized_array = np.zeros((kp_num, 128))
        if kp is None or des is None:
            return [], []
        if des.shape[0] >= kp_num:
            normalized_array = des[:kp_num]
        else:
            normalized_array[:des.shape[0]] = des
        feature = kmeans.predict(np.array(normalized_array, dtype=np.double))
        labels, distances = p1.knn_query(feature, k)
        return labels.squeeze(), distances.squeeze()
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_name = uploaded_file.name
        # 直接读取 bytes 为图像文件，会出现通道匹配错误，需要转换一下通道
        img = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

        st.header('Selected image')
        st.image(img)

        if st.button('Query'):
            t = time.time()
            labels, distances = query_image(bytes_data)
            end_t = time.time()
            st.write(f'Searched in {(end_t - t):.3f} s')
            if calc_ap is not None:
                search_results = []
                for label in labels:
                    search_results.append(img_map.loc[label, 'File'])
                ap = calc_ap(search_results, file_name)
                aps.append(ap)
                st.write(f'mAP = {calc_map(aps)}')
                st.write(f'AP = {ap}')
            column_num = 2
            cols = st.columns(column_num)
            for (idx, (label, dist)) in enumerate(zip(labels, distances)):
                image_name = img_map.loc[label, 'File']
                caption = f'{idx + 1}, {image_name}, dist={dist:.5f}'
                res_image_path = os.path.join(image_path, image_name)
                if idx % column_num == 0:
                    with cols[0]:
                        st.image(res_image_path, caption=caption)
                elif idx % column_num == 1:
                    with cols[1]:
                        st.image(res_image_path, caption=caption)
