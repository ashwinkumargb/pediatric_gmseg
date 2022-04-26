#!/bin/bash
#SBATCH --mail-user=ashwin.kumar@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00         # 12 hours
#SBATCH --mem=30G               # 10 GB RAM 
#SBATCH --output=gpu-job.log    # output file
#SBATCH --partition=pascal     # low-latency RoCE network and 4 Titan X GPUs per node
#SBATCH --account=vuiis_masi_gpu_acc     # substitute appropriate group here
#SBATCH --gres=gpu:1            # single GPU job
#SBATCH --job-name=gm_model_pretrain

ml GCC/6.4.0-2.28 CUDA OpenMPI/2.1.1 Python/3.6.3 TensorFlow Keras matplotlib/2.1.0-Python-3.6.3 
source py36/bin/activate

# Pass python script to run
python gm_model.py

