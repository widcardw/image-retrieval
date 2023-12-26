import numpy as np
from towhee import pipe, ops
import hnswlib
# from flask import Flask, render_template
# import sys

# if len(sys.argv) != 2:
#     print("请输入图片路径")
#     sys.exit(1)

# app = Flask(__name__)

# image_path = sys.argv[1]

folder_path = 'static/pic1/'
print("创建管道")
p = (
    pipe.input('path')
    .map('path', 'img', ops.image_decode.cv2())
    .map('img', 'embedding', ops.image_text_embedding.clip(model_name="clip-vit-base-patch32", modality='image'))
    .map('embedding', 'embedding', lambda x:x/np.linalg.norm(x))  # 归一化操作
    .output('embedding')
)

res = p('dataset/oxbuild/images/xxx').get()
vec = res[0]

dim = 512

print("创建索引")
p1 = hnswlib.Index(space = 'l2', dim = dim)
print("读取索引文件")
p1.load_index("17126_pic.bin")

p1.set_ef(20)
print("开始查询")
labels, distances = p1.knn_query(vec, k = 100)
print(labels)

# 将查询结果对应到csv中的行，读取图片数据
# csv_file_path = "output.csv"
image_paths = []

# 将图片加入到image_paths列表中
for line_num in labels[0]:
    image_paths.append('pic1/' + str(line_num) + '.jpg')


# @app.route('/')
# def hello_world():  # put application's code here
#     return render_template('image_gallery.html', image_paths=image_paths) # 将图片传给html进行显示

# if __name__ == '__main__':
#     app.run()