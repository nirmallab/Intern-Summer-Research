import pandas as pd
original_224 = pd.read_csv('tile_information.csv') 

column_names = [str(i) for i in range(1, 197)]
column_names.insert(0,"region")
column_names.insert(1,"patches")
column_names.insert(2,"total")
column_names.insert(3,"logic_check_total")

final = pd.DataFrame(columns = column_names)

final.region = original_224.region
final.total = original_224.total

import ast

def str_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

# Apply the function to the appropriate columns
final['region'] = final['region'].apply(str_to_list)

for index in range(len(final)):
    regions_16x16 = []
    area = final.region[index]
    start_row, end_row = area[0]
    start_col, end_col = area[1]
    for i in range(start_row, start_row + 224, 16):
        for j in range(start_col, start_col + 224, 16):
            regions_16x16.append([(i, i + 16), (j, j + 16)])
    final["patches"][index] = regions_16x16
        
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

from collections import Counter

def check_list(lst, allowed_values):
    return all(item in allowed_values for item in lst)

for i in range(len(final)):
    patch_list = final['patches'][i]
    logic_check = 0
    for j in range(len(patch_list)):
        region = patch_list[j]
        cells_inside = useful_data[(region[0][0] <= useful_data['Y_centroid']) & (useful_data['Y_centroid'] < region[0][1]) & (region[1][0] <= useful_data['X_centroid']) & (useful_data['X_centroid'] < region[1][1])]
        logic_check+= len(cells_inside)
        cell_types = cells_inside['manual_leiden_edges_necrosis_muscle']
        # Create a Counter object
        cell_counts = Counter(cell_types)
        dict_cell = dict(cell_counts)
        if len(dict_cell)==0:
             final[str(j+1)][i] = 0 
        else:
            class_index = -1
            num_cells = sum(dict_cell.values())
            for key, value in dict_cell.items():
              if (value/num_cells) >= (2/3):
                class_index = key
            if class_index == -1:
              class_index = len(mapping) + 1 
            final[str(j+1)][i] = class_index
        
    final.logic_check_total[i] = logic_check
    
final.to_csv('patch_information.csv', index=False)