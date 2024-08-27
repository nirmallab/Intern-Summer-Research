How To Guide:

*Server Code*

ERISXdl Example Job Submission.md & GPU Usage Guide.md: Both these files offer information on how to interface with ERISXdl (Signing in ERIS, setting up container/podman, and running jobs)

-additional resources: https://rc.partners.org/kb/article/3719 & https://rc.partners.org/kb/article/3718

node.sh: This file was the bash script for requesting and setting up a node from the ERISXdl server (CPU, GPU, Time, Container, the Bash script to run, etc)

job.sh: This file was the bash script that the node.sh called to check gpu count and run a specified python file

erisxdl.dockerfile: This file was the docker file used to build the container pushed onto harbor specified in node.sh

*Preprocessing*

download_models.ipynb: This file is for downloading the three vision transformer models locally 

rescale.py: This file is a python file for rescaling the H&E image  

13_preprocess.py: This file is used to produce an h5 file that contains the labels of the each cell from provided data and corresponding centered images of the cell (as a numpy array) which will used by the fine tuning model code

make_balanced.py: Same description to the one above except the number of entries for each class was upper bounded by the class with median number of entries (eg. if the class with the median number of entries has 100 entries in the entire data then no class can have more than 100 entries)

*Fine Tuning*

mahm_script.py & micro_script.py: These files are for fine tuning the models from Mahmood’s lab (UNI) and Microsoft (Gigapath). The framework was set up with PyLightning but aside from freezing all the layers (except an added linear layer for classification) all the 
parameters used are just generic initializations

vit_cell_script.py: This file is for fine tuning the model from Google trained on natural images (google/vit-base-patch16-224). The framework was set up with HuggingFace Transformer library but aside from freezing all the layers (except an added linear layer for classification) all the parameters used are just generic initializations

*Extracting embeddings*

part1.py: This file is used to produce an h5 file that contains the images of the 224x224 regions in H&E with cells inside of them as well as a csv that contains the information about the region (total number of cells in region and number of each cell type in the region)

part2.py: This file is used to a csv that contains the information for each patch of the selected regions from part1, specifically the boundary locations of each patch and the class label that is assigned to the patch based on if ⅔ majority is achieved of a certain cell type within the patch

part3.py: This file is used to generate an h5 file of the embeddings for each patch, a csv that holds further information for each extracted patch embedding (patch coordinates, label,etc), and filters the regions further by eliminating those where a quarter or more of the image of the region is whitespace 

mlp.ipynb: This file is incomplete and was intended for building the MLPs that would take in the extracted embeddings and corresponding labels

-additional resources: https://github.com/pytorch/examples/blob/main/mnist/main.py & 
https://github.com/mahmoodlab/MAPS/blob/main/maps/cell_phenotyping/networks.py


