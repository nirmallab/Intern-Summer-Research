### Login to ERISXdl Server using SSH
```bash
ssh user_id@erisxdl.partners.org 
```
Replace `user_id` with your actual user ID to establish an SSH connection to the ERISXdl server.

### Login to Harbor
```bash
podman login https://erisxdl.partners.org/harbor/
```
Use the same user ID and password as your ERISXdl login.

### Pull Container Image from Nvidia  
```bash
podman pull nvcr.io/nvidia/pytorch:23.07-py3 
```
The URL `nvcr.io/nvidia/pytorch:23.07-py3` represents the Nvidia PyTorch container image. Adjust the tag and version according to your requirements.

### Find Container Image ID
```bash
podman images
```
After pulling the container image, use `podman images` to find the Image ID associated with the newly pulled image.

### Tag Image for Harbor 
```bash
podman tag image_id erisxdl.partners.org/bwh-comppath-img-g/pytorch:dummy
```
Replace `image_id` in the `podman tag` command with the actual Image ID obtained in the previous step. This command tags the pulled image for pushing to the Harbor registry.

### Push Image on Harbor 
```bash
podman push erisxdl.partners.org/bwh-comppath-img-g/pytorch:dummy
```
Ensure that you have the necessary permissions to push the container image to the Harbor registry. This command pushes the tagged image to the specified Harbor repository and tag. Adjust the repository and tag as needed.

### Sample Python Script to Run
Save the following script as gpu_check.py in your home directory on the ERISXdl Server.
```
import torch
import argparse
import time

def matrix_multiplication(matrix_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate random matrices on the specified device
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Perform matrix multiplication on the specified device
    start_time = time.time()
    result = torch.matmul(a, b)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Matrix Multiplication Result (Matrix Size: {matrix_size}) on {device}:")
    print(f"Time taken for matrix multiplication: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform matrix multiplication with specified matrix size.")
    parser.add_argument("--matrix_size", type=int, help="Size of the square matrices for multiplication")
    
    args = parser.parse_args()


    # Perform matrix multiplication on the specified device
    matrix_multiplication(args.matrix_size)

```

### Shell Script to Call Python Script
Save this code as job.sh in your home directory on the ERISXdl Server.
```
#!/bin/bash

cd ~/

# Set the hyperparameters for the python script
matrix_size=1000


# Run the Python script with the specified arguments
python gpu_check.py --matrix_size matrix_size
```

### Shell Script to Submit Job using SLURM
Save this code as run_job.sh in your home directory on the ERISXdl Server.
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
export KUBE_IMAGE=erisxdl.partners.org/bwh-comppath-img-g/pytorch:dummy 

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
### Submit Job
```bash
sbatch run_job.sh
```
Use the `sbatch` command to submit a job defined in the `run_job.sh` script. This command queues the job for execution on the cluster.

### View Job Status
```bash
squeue 
```
Check the status of submitted jobs using the `squeue` command. This command displays a list of jobs in the queue along with their status, such as running, pending, or completed. Use this information to monitor the progress of your job and identify any issues or delays in the execution queue.
