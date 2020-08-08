#!/bin/bash
#SBATCH --job-name=road_stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ./slurm_log/output_%j.txt
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=lntruong@cpp.edu

eval "$(conda shell.bash hook)"
conda activate py3

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "=========================================="

declare -a folderNames=()

if [ $# == 0 ]
then
	folderNames+=(`ls ./detections/`)
else
	for folder in "$@"
	do
		folderNames+=($folder)
	done
fi

for folder in "${folderNames[@]}"
do
    echo "=========================================="
	echo $folder
    srun python3 sample_3.py --gt="./groundtruths/" --dt="./detections/$folder/"
    echo "=========================================="	
done