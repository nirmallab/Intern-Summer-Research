import torch #gpu acceleration
device = "cpu"
if (torch.cuda.is_available()):
  device = "cuda"
print("device: " + device)

from torchvision import transforms
import timm
import os

model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
local_dir = "./mahmood/weights"  # directory to save the model weights
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)

import h5py
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
    return images #return the images

images_array = load_data('cell_images.h5')

import cv2
import numpy as np

not_white_indices = []

for i in range(len(images_array)):
    gray_scale = cv2.cvtColor(images_array[i], cv2.COLOR_BGR2GRAY) 
    _, threshold = cv2.threshold(gray_scale, 245 , 255, cv2.THRESH_BINARY)
    white_pixel_count = np.sum(threshold == 255)
    total_pixel_count = threshold.size
    ratio = white_pixel_count/total_pixel_count
    if (ratio < 0.25):
        not_white_indices.append(i)

images_array_not_white = [images_array[i] for i in not_white_indices]

import pandas as pd

#image_data = pd.DataFrame()

#image_data['images'] = list(images_array_not_white)

labels = pd.read_csv('patch_information.csv')

labels = labels.iloc[not_white_indices]
labels.reset_index(drop=True, inplace=True)

import ast

def str_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

# Apply the function to the appropriate columns
labels['patches'] = labels['patches'].apply(str_to_list)


preTransform = transforms.Compose([transforms.ToTensor()])

import torch
from PIL import Image 

mean = torch.zeros(3)
std = torch.zeros(3)
for i in range (0, len(images_array_not_white)):
    img = images_array_not_white[i]
    img = Image.fromarray(img)
    img = preTransform(img)
    mean += img.mean(dim=( 1, 2))
    std += img.std(dim=(0, 1, 2))

mean /= len(images_array_not_white)
std /= len(images_array_not_white)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean.numpy()), std=list(std.numpy())),
    ]
)

from PIL import Image

index = 0
embedding_values = []
embedding_df = []
for i in range(len(images_array_not_white)):
    image = images_array_not_white[i]
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(dim=0) 
    all_embeddings = model.forward_features(image).squeeze(0)
    for j in range(1,197):
        embedding = all_embeddings[j,:].detach().numpy() #get the embedding for the jth patch
        embedding_values.append(embedding)
        new_row = {'image_num': i, 'image_region': labels["region"][i], 'patch_num': j, 'patch_region': labels["patches"][i][j-1], 'label': labels[str(j)][i], 'embedding_index': index}
        embedding_df.append(new_row)
        index += 1
        
embedding_values = np.array(embedding_values)

images_array_not_white= np.array(images_array_not_white)

embedding_data = pd.DataFrame(embedding_df,columns=['image_num', 'image_region', 'patch_num', 'patch_region', 'label', 'embedding_index'])

import h5py

with h5py.File('no_white_images.h5', 'w') as f: # create a h5 file
    f.create_dataset('images', data=images_array_not_white) # store the images

with h5py.File('embedding_values.h5', 'w') as f: # create a h5 file
    f.create_dataset('embeddings', data=embedding_values) # store the images
    
embedding_data.to_csv('embedding_data.csv', index=False)

print(len(embedding_values) == len(embedding_data))
print(embedding_values)