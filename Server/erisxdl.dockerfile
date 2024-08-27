FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN pip install -U --no-cache-dir h5py timm transformers accelerate evaluate huggingface_hub lightning scikit-image