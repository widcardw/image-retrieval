import numpy as np
from towhee import pipe, ops, DataLoader
import os
import hnswlib
import time
# import csv
import pandas as pd


folder_path = 'dataset/oxbuild/images/'
print("创建管道")
p = (
    pipe.input('path')
    .map('path', 'img', ops.image_decode.cv2())
    .map('img', 'embedding', ops.image_text_embedding.clip(model_name="../../vector_database/project/clip-vit-base-patch32", modality='image'))
    .map('embedding', 'embedding', lambda x:x/np.linalg.norm(x))  # 归一化操作
    .output('embedding')
)


output_path = os.path.join('output', 'dataset/oxbuild')
my_hash = {}
i = 0
data = []


print("创建数据集")
start_time = time.time()
all_images = os.listdir(folder_path)
for file in all_images:
    res = p(folder_path + file).get()
    key = tuple(res[0])
    data.append(res[0])
    i = i + 1
end_time = time.time()
print(f"创建数据集用时:{end_time - start_time}")

dim = 512
num_elements = i
ids = np.arange(num_elements)

print("创建索引")
start_time = time.time()
p1 = hnswlib.Index(space = 'l2', dim = dim)
print("初始化索引")
p1.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
print("往索引中添加数据")
p1.add_items(data, ids)
end_time = time.time()
print(f"创建索引用时:{end_time - start_time}")

print("保存索引结构")
p1.save_index(os.path.join(output_path, "17126_pic.bin"))

out_csv = pd.DataFrame(enumerate(all_images), columns=['Index', 'File'])

out_csv.to_csv(os.path.join(output_path, 'output.csv'))


# 获取文件夹中的文件列表
# files = os.listdir(folder_path)



# 将文件名写入 CSV 文件
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Index', 'File'])  # 写入 CSV 文件的列名

#     for index, file_name in enumerate(files):
#         new_filename = f"{index}.jpg"
#         # 构建旧文件和新文件的完整路径
#         old_filepath = os.path.join(folder_path, file_name)
#         new_filepath = os.path.join(folder_path, new_filename)
#         # 重命名文件
#         os.rename(old_filepath, new_filepath)