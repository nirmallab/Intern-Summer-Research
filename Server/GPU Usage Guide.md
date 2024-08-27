# Quick Start Guide for ERISXdl GPU Cluster

## Overview

ERISXdl, short for ERIS Extreme Deep Learning, is a state-of-the-art GPU Cluster designed by ERIS Scientific Computing and powered by the NVIDIA DGX-1 for AI and deep learning-based research. The cluster consists of five nodes, each equipped with 8 NVIDIA Tesla V100 GPUs. ERISXdl brings together over 200 thousand CUDA cores, 25600 Tensor cores, 1280 GB of GPU memory, and 35TB of local storage for efficient data processing, enabling users to train larger models swiftly and conduct a broader range of experiments.

## Prerequisite for Access 

To gain access, please complete the form found at this [link](https://rc.partners.org/scic-cluster-account-request). Once submitted, you will receive an email confirmation acknowledging receipt of your request. Another email containing your account information will be sent once your account is active. Please allow up to a day for this process to be completed. 

After that you should reach out to Richard Kenny (RKENNY@PARTNERS.ORG), cc Tim (tjanicki@bwh.harvard.edu), Georg and Faisal, asking to be added to ERISXdl and also the group related to your lab (e.g., bwh-comp-path-img), which will give you access to the nodes as well as the cloud storage (Briefcase) that belong to your lab. 

## Logging in on ERISXdl

The ERISXdl Platform is accessed via the Linux command-line interface tool ssh so that some familiarity with Linux terminal commands is required. More precisely, users can ssh from within the Mass General Brigham network or via VPN if offsite. There are three login nodes (ERISXDL1, ERISXDL2, and ERISXDL3), and the load balancer will automatically find the available login node.

```
ssh your_bwh_id@erisxdl.partners.org
```
Replace **your_bwh_id** with your BWH id and enter your BWH password when prompted. For additional log-in options, refer to [this link](https://rc.partners.org/kb/computational-resources/erisxdl?article=3654).

If you encounter issues such as command execution failure or unusually delayed server responses, then switch to another login node without logging out from your current session. You can accomplish this by executing one of the following commands: 
```
ssh erisxdl1
ssh erisxdl2
ssh erisxdl3
```
### Home Directory

Your home directory serves as the designated space for storing code and scripts for job execution. It is important to note that due to limited space, **it is advisable to refrain from storing large datasets here**. You have the freedom to structure your home directory in a way that best suits your workflow, allowing you to create folders and sub-folders as needed. 

To view the contents of your home directory on Linux, use the following command:

```
ls ~
```

### Lab Cloud Storage (Briefcase)

Lab Cloud Storage, also referred to as Briefcase, provides a dedicated space to house large datasets associated with your scheduled jobs. Please be aware that while the storage capacity is generous (5+ TB), it is not intended for long-term or archival storage. It is strongly advised to clear out data associated with completed jobs or projects on a regular basis. 

For enhanced organization, **it is highly recommended to create a folder under your name and store all data related to your jobs and projects within it**. This practice facilitates easy tracking of ownership for each set of data. 

To access the contents of the Lab Cloud Storage directory on Linux, you can utilize one of the following commands: 

```
ls /data/
ls /data/bwh-comppath-full
ls /data/bwh-comppath-img
```

One can mount the cloud storage (Briefcase) drive by using the following command:
```
sudo apt-get install sshfs
mkdir ~/erisxdl-data
sudo sshfs -o allow_other your_bwh_id@erisonexf.partners.org:/data/bwh-comppath-img/ ~/erisxdl-data
```

## Managing Images and Containers with Podman 

An image serves as a template bundling all the necessary dependencies for running applications, while containers are the active instances of these images, capable of executing applications with their own data and state. Podman, an efficient tool, is employed to manage images and containers on the login nodes of ERISXdl. 

### Downloading an Image 

To get started, acquire a suitable pre-built image from a public registry such as nvcr.io.nvidia or from Harbor. Use the following command to download an image from the desired source: 
```
podman pull registry_path/image_name:tag 
```
For example: 
```
podman pull nvcr.io/nvidia/pytorch:23.07-py3 
```
You can verify the successful download of the image using the following command. It will display details including the repository, tag, image ID, and size of all available images in your home directory:  
```
podman images 
```
**Note**: Avoid accumulating multiple images, as they are stored on a shared storage disk with limited capacity. It is recommended to delete unused images using the following command: 
```
podman rmi -f image_id 
```
Make sure to replace **image_id** with the actual ID of the image you want to remove. 

### Customizing the Image 

It is important to remember that GPU nodes lack direct internet access. Therefore, containers and code should be prepared or updated before job submission. It's essential to ensure that all the necessary libraries for your code/application are available in the image. To do this, you can run the image as a container using the following command. This command will also map your home directory folder to the home folder in the running container:  
```
podman run --mount type=bind,src=/PHShome/your_bwh_id/,target=/home -it image_id /bin/bash 
```
In the running container, install any missing packages. Once you've finished customizing the image container, use the 'exit' command to leave the container. If you made changes in the container, you can obtain the updated container ID using the following command: 
```
podman ps -a 
```
Use the updated container ID to commit the changes to the existing image or save it as a modified version with a different tag: 
```
podman commit container_id registry_path/image_name:tag 
```
### Pushing to Harbor 

Once you've completed customizing the image, the next step is to push it to the Harbor registry to ensure smooth execution on GPU nodes. First, log in to the Harbor registry using the following command: 
```
podman login erisxdl.partners.org 
```
Use BWH user_id and password for login.  

You also need to tag the container image for the Harbor registry. Use the following command to do so: 
```
podman tag image_id erisxdl.partners.org/PAS_Group_Name/image_name:tag 
```
Replace PAS_Group_Name with the lowercase name of the Briefcase of your lab (e.g., bwh-comppath-img-g), and ensure you specify the correct image_name and tag. Finally, push the container image to the Harbor registry with the following command: 
```
podman push erisxdl.partners.org/ PAS_Group_Name/image_name:tag 
```
Please be aware that the Harbor registry has limited storage space of 50GB for each group. Therefore, exercise caution when pushing container images to the Harbor registry. Additionally, it is recommended to include your initials in the tag name to easily track images associated with your projects. 

### Podman Settings 

On ERISXdl there are three login nodes, erisxdl1, erisxdl2 and erisxdl3 and where each will contain differing collections of locally-stored images. In order to ensure the user has access to the images on a given node please locate the following file in the home directory:  
```
~/.config/containers/storage.conf 
```
Make the following change using your favorite text editor: 
```
graphroot = "/erisxdl/local/storage/bwh_id" 
```
Replace bwh_id with your BWH userid. This allows Podman to function on all nodes, even in case of node failure. 

## Running Jobs on ERISXdl 

Slurm is the scheduler responsible for allocating resources to submitted jobs on ERISXdl. It is essential to submit all jobs through the SLURM scheduler system. ERISXdl cluster have partitions with different configurations, such as varying numbers of nodes, memory capacities, and GPU availability. Users can select the appropriate partition for their jobs, depending on the specific requirements of their computational tasks. To view the available partitions, use the following command: 
```
sinfo 
```
Please note that, with the exception of the Basic partition, all other partitions require group and fund number registration for job submission. 

For additional info on a specific partition, execute command: 
```
sinfo --long -p partition_name 
```
There are several GPU nodes that accept jobs from all the partitions. The state of these 5 nodes dgx-[1-5] at any given time can be inspected with the following: 
```
scontrol show nodes 
```
To submit a job, write a bash script with the SBATCH flags specified at the top of the file. Following is an example job script: 
```
#!/bin/bash 

#SBATCH --partition=Basic 
#SBATCH --job-name=test_job 
#SBATCH --gpus=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2 
#SBATCH --time=00:10:00 
#SBATCH --mem=16G 
#SBATCH --output=logs.%j 
#SBATCH --error=errors.%j 

# Set the docker container image to be used in the job runtime. 
export KUBE_IMAGE=erisxdl.partners.org/bwh-comppath-img-g/pytorch:ms-mm-toad 

# Set the script to be run within the specified container - this MUST be a separate script 
export KUBE_SCRIPT=$SLURM_SUBMIT_DIR/job.sh

# Ensure job.sh is executable 
chmod a+x  $SLURM_SUBMIT_DIR/job.sh 

# (Optional) Make a copy of the job.sh by adding job_id in file name for future reference
cp $SLURM_SUBMIT_DIR/job.sh $SLURM_SUBMIT_DIR/job_$SLURM_JOB_ID.sh 

# Define working directory to use
export KUBE_DATA_VOLUME=/data/bwh-comppath-img/

# Users can also set the following variable to change the timeout in seconds. It’s 600 by default, but might be useful to change for testing. 
export KUBE_INIT_TIMEOUT=300 

# Required wrapper script. This must be included at the end of the job submission script. 
# This wrapper script mounts /data, and your /PHShome directory into the container  
srun  /data/erisxdl/kube-slurm/wrappers/kube-slurm-custom-image-job.sh  
```
To submit a job script: 
```
sbatch example_script.sh 
```
After submitting your jobs, always check that your jobs have been submitted successfully. To check the job status use following command: 
```
squeue 
```
View more verbose job status 
```
squeue -j job_ID 
```
Check job in detail: 
```
scontrol show job job_ID 
```
Check this [link](https://rc.partners.org/kb/computational-resources/erisxdl?article=3719) for a more detailed guide for running jobs on the ERISXdl cluster. 
