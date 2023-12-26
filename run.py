import streamlit as st
from PIL import Image
import numpy as np
import hnswlib
from towhee import pipe, ops
import pandas as pd
import os
import time

p = (
    pipe.input('path')
        .map('path', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_text_embedding.clip(model_name="../../vector_database/project/clip-vit-base-patch32", modality='image'))
        .map('embedding', 'embedding', lambda x:x/np.linalg.norm(x))  # 归一化操作
        .output('embedding')
)

img_map = pd.read_csv('output/dataset/oxbuild/output.csv')
image_path = 'dataset/oxbuild/images/'

dim = 512

p1 = hnswlib.Index(space = 'l2', dim = dim)
p1.load_index("output/dataset/oxbuild/17126_pic.bin")
p1.set_ef(20)

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

k = st.slider('Result count', min_value=1, max_value=15, step=1, value=5)

def query_image(img: bytes):
    res = p(img).get()
    vec = res[0]
    labels, distances = p1.knn_query(vec, k)
    return labels.squeeze(), distances.squeeze()


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img_type = uploaded_file.type

    if st.button('Query'):
        t = time.time()
        labels, distances = query_image(bytes_data)
        end_t = time.time()
        st.write(f'Searched in {(end_t - t):.3f} s')
        for (label, dist) in zip(labels, distances):
            image_name = img_map.loc[label, 'File']
            st.image(
                os.path.join(image_path, image_name),
                caption=f'{image_name}, {dist:.5f}'
            )

