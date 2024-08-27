import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# Load the RGB image
image = io.imread('necrosis_muscle.tif')

# Initialize an array to hold the crops
crops = []
regions = []

# Loop through the image to extract 224x224 crops
for i in range(0, image.shape[0], 224):
    for j in range(0, image.shape[1], 224):
        crop = image[i:i+224, j:j+224]
        if crop.shape[0] == 224 and crop.shape[1] == 224:
            crops.append(crop)
            region= [(i,i+224),(j,j+224)]
            regions.append(region)

import pandas as pd

df = pd.read_csv('phenotype.csv') # read the csv file

useful_data = df[["X_centroid", "Y_centroid","manual_leiden_edges_necrosis_muscle"]] # select the columns that we need
#useful_data["X_centroid_rescaled"] = useful_data["X_centroid"] * 0.65 # rescale the X_centroid
#useful_data["Y_centroid_rescaled"] = useful_data["Y_centroid"] * 0.65 # rescale the X_centroid

useful_data.head()
useful_data.reset_index(drop=True, inplace=True)

new_mapping_df = useful_data[useful_data.manual_leiden_edges_necrosis_muscle != "excluded"] # remove the rows that are not useful
new_mapping_df = new_mapping_df[new_mapping_df.manual_leiden_edges_necrosis_muscle != "other immune cells"] # remove the rows that are not useful
new_mapping_df = new_mapping_df[new_mapping_df.manual_leiden_edges_necrosis_muscle != "edges"] # remove the rows that are not useful


mapping =  dict((v, i) for i, v in enumerate(new_mapping_df.manual_leiden_edges_necrosis_muscle.unique(),1))

mapping["excluded"] = len(mapping) + 1
mapping["other immune cells"] = len(mapping) + 1
mapping["edges"] = len(mapping) + 1

print(mapping)

useful_data['manual_leiden_edges_necrosis_muscle'] = useful_data['manual_leiden_edges_necrosis_muscle'].map(mapping)
useful_data.head()

#make a new dataframe
region_df = pd.DataFrame(columns = ['region','total', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15','16'])

region_df['region'] = regions

for i in range(len(region_df)):
    region = region_df['region'][i]
    cells_inside = useful_data[(region[0][0] <= useful_data['Y_centroid']) & (useful_data['Y_centroid'] < region[0][1]) & (region[1][0] <= useful_data['X_centroid']) & (useful_data['X_centroid'] < region[1][1])]
    region_df['total'][i] = len(cells_inside)
    for j in range(1,17):
        region_df[str(j)][i] = len(cells_inside[cells_inside['manual_leiden_edges_necrosis_muscle'] == j])
        
region_df = region_df[region_df.total != 0]

indicies = list(region_df.index)

cell_images = [crops[i] for i in indicies]

cell_images = np.array(cell_images)

region_df.reset_index(drop=True, inplace=True)

region_df.to_csv('tile_information.csv', index=False)

import h5py

with h5py.File('cell_images.h5', 'w') as f: # create a h5 file
    f.create_dataset('images', data=cell_images) # store the images