import pandas as pd
original_224 = pd.read_csv('tile_information.csv') # read the csv file

column_names = [str(i) for i in range(1, 197)] # create the column names for the patches
column_names.insert(0,"region") # add the region column
column_names.insert(1,"patches") # add the patches column 
column_names.insert(2,"total") # add the total column
column_names.insert(3,"logic_check_total") # add the logic_check_total column

final = pd.DataFrame(columns = column_names) # create the dataframe

final.region = original_224.region # copy the region column
final.total = original_224.total  # copy the total column

import ast

def str_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

# Apply the function to the appropriate columns
final['region'] = final['region'].apply(str_to_list) # convert the string to a list

for index in range(len(final)): # iterate over the rows/regions
    regions_16x16 = [] # create an empty list to store the 16x16 patches
    area = final.region[index] # get the region
    start_row, end_row = area[0] # get the start and end row
    start_col, end_col = area[1] # get the start and end column
    for i in range(start_row, start_row + 224, 16): # iterate over the rows
        for j in range(start_col, start_col + 224, 16): # iterate over the columns
            regions_16x16.append([(i, i + 16), (j, j + 16)]) # append the 16x16 patch
    final["patches"][index] = regions_16x16 # add the 16x16 patches to the dataframe
        
import pandas as pd

df = pd.read_csv('phenotype.csv') # read the csv file

useful_data = df[["X_centroid", "Y_centroid","manual_leiden_edges_necrosis_muscle"]] # select the columns that we need

useful_data.head()
useful_data.reset_index(drop=True, inplace=True) # reset the index

#create a new mapping (same as part1)
new_mapping_df = useful_data[useful_data.manual_leiden_edges_necrosis_muscle != "excluded"] 
new_mapping_df = new_mapping_df[new_mapping_df.manual_leiden_edges_necrosis_muscle != "other immune cells"] 
new_mapping_df = new_mapping_df[new_mapping_df.manual_leiden_edges_necrosis_muscle != "edges"] 


mapping =  dict((v, i) for i, v in enumerate(new_mapping_df.manual_leiden_edges_necrosis_muscle.unique(),1))

mapping["excluded"] = len(mapping) + 1
mapping["other immune cells"] = len(mapping) + 1
mapping["edges"] = len(mapping) + 1

print(mapping)

useful_data['manual_leiden_edges_necrosis_muscle'] = useful_data['manual_leiden_edges_necrosis_muscle'].map(mapping) # map the values
useful_data.head()

from collections import Counter

for i in range(len(final)): # iterate over the rows/regions
    patch_list = final['patches'][i] # get the patches
    logic_check = 0 # create a variable to store the logic check for each region
    for j in range(len(patch_list)): # iterate over the patches
        region = patch_list[j] # get the patch
        # Select the cells inside the patch
        cells_inside = useful_data[(region[0][0] <= useful_data['Y_centroid']) & (useful_data['Y_centroid'] < region[0][1]) & (region[1][0] <= useful_data['X_centroid']) & (useful_data['X_centroid'] < region[1][1])]
        # Count the number of cells inside the patch
        logic_check+= len(cells_inside)
        cell_types = cells_inside['manual_leiden_edges_necrosis_muscle']  # Get the cell types
        cell_counts = Counter(cell_types) # Create a Counter object
        dict_cell = dict(cell_counts) # convert the Counter object to a dictionary
        if len(dict_cell)==0: # if there are no cells in the patch
             final[str(j+1)][i] = 0  # assign 0
        else: # if there are cells in the patch
            class_index = -1 # create a variable to store the class index
            num_cells = sum(dict_cell.values()) # get the number of cells
            for key, value in dict_cell.items(): # iterate over the cell types
              if (value/num_cells) >= (2/3): # if the percentage of the cell type is greater than 2/3
                class_index = key # assign the class index
            if class_index == -1: #if no class index is assigned
              class_index = len(mapping) + 1 # assign the class index to the last class + 1 (usually 17 in this case assuming 16 classes) 
            final[str(j+1)][i] = class_index # assign the class index to the patch
        
    final.logic_check_total[i] = logic_check # assign the logic check to the region
    
final.to_csv('patch_information.csv', index=False) # save the dataframe to a csv file
