#!/bin/bash
#SBATCH --job-name=ml_elf
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --cluster=faculty
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --output=S.out

module purge
module load gcc/11.2.0
module load openmpi/4.1.1
module load pytorch/1.13.1-CUDA-11.8.0

ulimit -s unlimited
export I_MPI_PMI_LIBRARY=/opt/software/slurm/lib64/libpmi.so
export pyexec=/projects/academic/ezurek/xiaoyu/elf

pres="S"
workdir="./${pres}_tmp"
logdir="./${pres}_log"
mkdir -p "${workdir}"
mkdir -p "${logdir}"

for i in $(seq 1 1 100); do

  echo "============================================================"
  echo "Repeat ${i}"
  echo "Selection file: ./sel/select_${i}.txt"
  echo "Temporary workdir: ${workdir}"
  echo "============================================================"

  # Clean previous repeat files
  rm -rf "${workdir}"
  mkdir -p "${workdir}"

  # 1. Make training dataset from selected binary structures
  srun python "${pyexec}/make_training_data_variable_elements.py" \
    $(cat "./sel/select_${i}.txt") \
    --num-samples-per-file 100 \
    --species-order H Mg S \
    --r-cut 3.0 \
    --n-radial 10 \
    --lmax 2 \
    --sigma-r 0.35 \
    --out "${workdir}/train.npz" \
    --meta-out "${workdir}/train_meta.json" \
    --log-out "${workdir}/make_training_data.log"

  # 2. Train checkpoint; overwritten every loop
  srun python "${pyexec}/train_variable_elements.py" \
    "${workdir}/train.npz" \
    --meta "${workdir}/train_meta.json" \
    --outdir "${workdir}/train_mlp" \
    --hidden-dims 512 256 128 \
    --batch-size 2048 \
    --epochs 200 \
    --lr 2e-3 \
    --weight-decay 1e-6 \
    --huber-delta 0.05 \
    --patience 20 \
    --log-out "${workdir}/train.log"

  # 3. Evaluate true ternary target; temporary raw outputs
  srun python "${pyexec}/evaluate_predict_variable_elements.py" \
    ./S12/*/ELFCAR \
    --checkpoint "${workdir}/train_mlp/best_model.pt" \
    --outdir "${workdir}/eval_Mg4S16H48" \
    --num-samples-per-file 100 \
    --log-out "${workdir}/eval_Mg4S16H48.log"

  # 4. Keep only compact logs / metrics for this repeat
  cp "${workdir}/train.log" "./${logdir}/train_${i}.log"
  cp "${workdir}/eval_Mg4S16H48.log" "./${logdir}/eval_${i}.log"

  if [ -f "${workdir}/eval_Mg4S16H48/evaluation_metrics.json" ]; then
    cp "${workdir}/eval_Mg4S16H48/evaluation_metrics.json" "./${logdir}/evaluation_metrics_${i}.json"
  fi

  if [ -f "${workdir}/train_mlp/split_and_metrics.json" ]; then
    cp "${workdir}/train_mlp/split_and_metrics.json" "./${logdir}/train_metrics_${i}.json"
  fi

  echo "Finished repeat ${i}"

done

rm -rf "${workdir}"
