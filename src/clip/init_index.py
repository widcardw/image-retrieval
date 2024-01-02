import numpy as np
from towhee import pipe, ops, DataLoader
import os
import hnswlib
import time
# import csv
import pandas as pd
from tqdm import tqdm
from constants import model_name


def init_index(dataset_folder: str, model_name = model_name):
    '''
    Images of the dataset should be put in {dataset_folder}/images
    If {model_name} is not provided, use `clip-vit-base-path32` by default

    However, we lose connection with huggingface, so I should download the model and turn to relative path.
    '''
    folder_path = os.path.join(dataset_folder, 'images')
    if not os.path.exists(folder_path):
        raise Exception(f'Images should be put in {folder_path}')
    print("Creating pipeline")
    print('Using model', model_name)
    # Build pipeline with towhee
    p = (
        pipe.input('path')
        .map('path', 'img', ops.image_decode.cv2())
        .map('img', 'embedding', ops.image_text_embedding.clip(model_name=model_name, modality='image'))
        .map('embedding', 'embedding', lambda x: x / np.linalg.norm(x))
        .output('embedding')
    )

    output_path = os.path.join('output', dataset_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    i = 0
    data = []

    print("Creating dataset")
    start_time = time.time()
    all_images = os.listdir(folder_path)
    for file in tqdm(all_images):
        res = p(os.path.join(folder_path, file)).get()
        data.append(res[0])
        i = i + 1
    end_time = time.time()
    print(f"Dataset created in: {end_time - start_time}s")

    # The output dim of CLIP is 512
    dim = 512
    num_elements = i
    ids = np.arange(num_elements)

    print("Creating index")
    start_time = time.time()
    # Build hnsw index with euclidean distance
    p1 = hnswlib.Index(space = 'l2', dim = dim)
    print("Init index")
    p1.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
    print("Inserting data")
    p1.add_items(data, ids)
    end_time = time.time()
    print(f"Index created in: {end_time - start_time}s")

    index_file_path = os.path.join(output_path, "clip.bin")
    indexed_image_list_file_path = os.path.join(output_path, 'clip.csv')
    print('Save Index at', index_file_path)
    p1.save_index(index_file_path)

    # Save the map from image id to image path
    out_df = pd.DataFrame(enumerate(all_images), columns=['Index', 'File'])
    out_df.to_csv(indexed_image_list_file_path)
