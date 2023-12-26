import torch
from PIL import Image
from torchvision import transforms
from typing import List
import numpy as np

def get_image_feature(img: Image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img)
    # input_batch = input_tensor.unsqueeze(0)
    input_batch = input_tensor
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    features = output.cpu().squeeze().detach().numpy()
    return features


import os
import pandas as pd

# def get_image_list(dataset_path: str):
#     image_path = os.path.join(dataset_path, 'images')
#     gt_path = os.path.join(dataset_path, 'gt.csv')
#     if not os.path.isdir(image_path):
#         raise Exception('Image set not exist!')
#     if not os.path.exists(gt_path):
#         raise Exception('GT not exist!')
#     gt = pd.read_csv(gt_path)
#     gt = gt[gt['label'] > 0]
#     image_list = []
#     for _, row in gt.iterrows():
#         file_name = row['name'] + '.jpg'
#         if os.path.exists(os.path.join(image_path, file_name)):
#             image_list.append(file_name)
#     return image_list

def get_image_list(dataset_path: str):
    files = os.listdir(os.path.join(dataset_path, 'images'))
    return files


from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path


def save_features(dataset_path: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_paths = list(map(lambda x: os.path.join(dataset_path, 'images', x), get_image_list(dataset_path)))
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')
    all_features = []
    all_image_paths = []
    with torch.no_grad():
        for images, image_paths in dataloader:
            if torch.cuda.is_available():
                images = images.to('cuda')
            features = model(images)
            all_features.append(features)
            all_image_paths.extend(image_paths)
    all_features: np.ndarray = torch.cat(all_features, dim=0).cpu().squeeze().detach().numpy()
    output_path = os.path.join('output', dataset_path)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'features.pkl'), 'wb') as fout:
        all_features.dump(fout)
    with open(os.path.join(output_path, 'image_list.txt'), 'w') as fout:
        fout.writelines(map(lambda x: x + '\n', all_image_paths))
    return all_image_paths


if __name__ == '__main__':
    save_features('dataset/oxbuild')
