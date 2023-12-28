import streamlit as st

import numpy as np
import hnswlib
import pandas as pd
import os
import time
import cv2

from src.sift.constants import kp_num, dim

sift = cv2.SIFT_create(kp_num)

def subpage_sift(dataset_name: str, title = ''):
    img_map = pd.read_csv(f'output/dataset/{dataset_name}/output.csv')
    image_path = f'dataset/{dataset_name}/images/'

    p1 = hnswlib.Index(space='l2', dim=dim)
    p1.load_index(f'output/dataset/{dataset_name}/sift.bin')
    p1.set_ef(20)

    st.title(title if title.strip() != '' else dataset_name)
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    k = st.slider('Result count', min_value=1, max_value=15, step=1, value=5)

    def query_image(img_bytes: bytes):
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        feature: np.ndarray = des.flatten()
        labels, distances = p1.knn_query(feature, k)
        return labels.squeeze(), distances.squeeze()
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        if st.button('Query'):
            t = time.time()
            labels, distances = query_image(bytes_data)
            end_t = time.time()
            st.write(f'Searched in {(end_t - t):.3f} s')
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
