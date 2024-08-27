import torch #gpu acceleration
device = "cpu"
if (torch.cuda.is_available()):
  device = "cuda"
print("device: " + device)

from torchvision import transforms
import timm
import os

model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True) #create model architecture
local_dir = "./mahmood/weights"  # directory where model is saved
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True) #load weights into model

import h5py
def load_data(file_path): #function to read images (as numpy arrays from) the h5 file
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
    return images #return the images

images_array = load_data('cell_images.h5')

import cv2
import numpy as np

not_white_indices = [] #list to store indices of images that are not have less than 25% whitespace

for i in range(len(images_array)): #iterate through all images
    gray_scale = cv2.cvtColor(images_array[i], cv2.COLOR_BGR2GRAY)  #convert image to grayscale
    _, threshold = cv2.threshold(gray_scale, 245 , 255, cv2.THRESH_BINARY) #threshold the image
    white_pixel_count = np.sum(threshold == 255) #count the number of white pixels
    total_pixel_count = threshold.size #count the total number of pixels
    ratio = white_pixel_count/total_pixel_count #calculate the ratio of white pixels to total pixels
    if (ratio < 0.25): #if the ratio is less than 25%
        not_white_indices.append(i) #add the index to the list

images_array_not_white = [images_array[i] for i in not_white_indices] #get the images that are not mostly white

import pandas as pd

labels = pd.read_csv('patch_information.csv') #read the labels csv file

labels = labels.iloc[not_white_indices] #get the labels for the images that are not mostly white

labels.reset_index(drop=True, inplace=True) #reset the index

import ast

def str_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

# Apply the function to the appropriate columns
labels['patches'] = labels['patches'].apply(str_to_list) #convert the string representation of a list to a list

preTransform = transforms.Compose([transforms.ToTensor()]) #transform to convert image to tensor

#use code to find mean and standard deviation values for filtered images
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
embedding_values = [] #list to store the embeddings
embedding_df = [] #list to store the embedding data
for i in range(len(images_array_not_white)): #iterate through all filtered images
    image = images_array_not_white[i] #get the ith image
    image = Image.fromarray(image) #convert the image to a PIL image
    image = transform(image).unsqueeze(dim=0)  #apply the transform to the image
    all_embeddings = model.forward_features(image).squeeze(0) #get the embeddings for all patches in the image
    for j in range(1,197): #iterate through all patches
        embedding = all_embeddings[j,:].detach().numpy() #get the embedding for the jth patch
        embedding_values.append(embedding) #add the embedding to the list
        new_row = {'image_num': i, 'image_region': labels["region"][i], 'patch_num': j, 'patch_region': labels["patches"][i][j-1], 'label': labels[str(j)][i], 'embedding_index': index} #create a new row for the embedding data
        embedding_df.append(new_row)  #add the row to the list
        index += 1
        
embedding_values = np.array(embedding_values) #convert the list of embeddings to a numpy array

images_array_not_white= np.array(images_array_not_white) #convert the list of images to a numpy array

embedding_data = pd.DataFrame(embedding_df,columns=['image_num', 'image_region', 'patch_num', 'patch_region', 'label', 'embedding_index']) #create a dataframe from the embedding data

import h5py

with h5py.File('no_white_images.h5', 'w') as f: # create a h5 file
    f.create_dataset('images', data=images_array_not_white) # store the images

with h5py.File('embedding_values.h5', 'w') as f: # create a h5 file
    f.create_dataset('embeddings', data=embedding_values) # store the embeddings
    
embedding_data.to_csv('embedding_data.csv', index=False) #save the embedding data to a csv file
