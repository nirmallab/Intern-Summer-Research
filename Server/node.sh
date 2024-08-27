#!/bin/bash 

#specify duration, ram memory size, gpus, nodes, location of error & log file, name of job, and partition which is tied with duration

#SBATCH --partition=Medium
#SBATCH --job-name=large_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs.%j
#SBATCH --error=errors.%j

# Set the docker container image to be used in the job runtime.
export KUBE_IMAGE=erisxdl.partners.org/bwh-nirmallab-eris1-g/yag1/erisxdl:with-scikit_image

# Set the script to be run within the specified container - this MUST be a separate script
export KUBE_SCRIPT=$SLURM_SUBMIT_DIR/job.sh

# Ensure job.sh is executable
chmod a+x  $SLURM_SUBMIT_DIR/job.sh

# Define working directory to use
export KUBE_DATA_VOLUME=/data/

# Required wrapper script. This must be included at the end of the job submission script. 
# This wrapper script mounts /data, and your /PHShome directory into the container  
srun /data/erisxdl/kube-slurm/wrappers/kube-slurm-custom-image-job.sh
