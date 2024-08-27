FROM nvcr.io/nvidia/pytorch:22.12-py3 
#Pull an existing nvidia container that uses CUDA version compatiable with the CUDA drivers on server (can check with nvidia-smi) & last I checked the drivers didn't support any 12.x version

RUN pip install -U --no-cache-dir h5py timm transformers accelerate evaluate huggingface_hub lightning scikit-image 
#Download python libraries
