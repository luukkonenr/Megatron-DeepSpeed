#!/bin/bash

#SBATCH --job-name=viking_13_wgqa
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH -p dev-g
##SBATCH -p standard-g
#SBATCH -t 00-00:30:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000086
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p workdir
wd=$(realpath workdir)


# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup

CONTAINER="/pfs/lustrep4/scratch/project_462000319/rluukkon/flash-attn-test-2"
SING_BIND="/scratch/project_462000319,/flash/project_462000319"


# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs

LEARNING_RATE=1e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s "$SLURM_JOB_ID.out" logs/latest.out
ln -f -s "$SLURM_JOB_ID.err" logs/latest.err

mkdir -p tensorboard/
mkdir -p checkpoints/
CHECKPOINT_PATH=checkpoints/
TENSORBOARD_PATH="tensorboard/$SLURM_JOB_ID"

# Data

export CUDA_DEVICE_MAX_CONNECTIONS=1

EVAL_INTERVAL=5000
EVAL_STEPS=100

DATA_PATH="0.07370182629 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/finnish 0.3302641761 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/slimpajama 0.330497442 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/starcoderdata 0.08367352788 /scratch/project_462000319/viking_preprocessed_data/merged_datasets/nordic-en-xling-combined 0.002361170146 /scratch/project_462000319/viking_preprocessed_data/small_files/train-books_text_document 0.05157063372 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-da-train_text_document 0.004054463623 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-is-train_text_document 0.08052558051 /scratch/project_462000319/viking_preprocessed_data/nordics/mc4-sv-train_text_document 0.04188033719 /scratch/project_462000319/viking_preprocessed_data/nordics/nor_all_combined_text_document 0.001470842506 /scratch/project_462000319/viking_preprocessed_data/small_files/natural_instruct_train_text_document"
MERGES=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/merges.txt
VOCAB=/scratch/project_462000319/tokenizers/nordic_tokenizer_131072/vocab.json


PP_SIZE=4
TP_SIZE=2

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512

# export MEMORY_OPT_ALLREDUCE_SIZE=2500000000
export MEMORY_OPT_ALLREDUCE_SIZE=1500000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

# TRAIN_SAMPLES=976_562_500
TRAIN_SAMPLES=488_281_250
TRAIN_SAMPLES=${TRAIN_SAMPLES//_}    # drop "_" for bash math
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES/100))


# 13B params without swiglu 
NLAYERS=40
NHIDDEN=5120
NHEADS=40
FFN_HIDDEN_SIZE=$((4*NHIDDEN))
SEQ_LEN=4096


# NLAYERS=48
# NHIDDEN=5120
# NHEADS=40
# FFN_HIDDEN_SIZE=18432
# SEQ_LEN=4096

SAVE_INTERVAL=2500

EVAL_INTERVAL=60000
EVAL_STEPS=100

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr $LEARNING_RATE \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --num-key-value-heads 4 \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --init-method-std 0.0048 \
    --bf16 \
    --seed 42 \
    --use-flash-attn \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-gradient-accumulation-fusion \
    --make-vocab-size-divisible-by 128 \
    --use-rotary-position-embeddings \
    --swiglu \
    $OPTIMIZER_ARGS \
    "
    # --num-layers-per-virtual-pipeline-stage 5 \
    # --ds-sequence-parallel-size 4 \

    # --no-pipeline-parallel \
    # --recompute-method uniform \
    # --checkpoint-activations \
    # --llama-MLP-projection \
    # --sync-tp-duplicated-parameters \
    # --embed-layernorm \

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL    \
    --eval-iters $EVAL_STEPS \
    "
    # --tensorboard-dir $TENSORBOARD_PATH \
    # --tensorboard-queue-size 5 \
    # --log-timers-to-tensorboard \
    # --log-batch-size-to-tensorboard \
    # --log-validation-ppl-to-tensorboard \
    
ZERO_STAGE=0

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$SLURM_JOB_ID.json"

cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "bf16": {
        "enabled": true
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE \
    "
    # --train-weighted-split-paths-path $TRAIN_DATA_PATH \
    # --valid-weighted-split-paths-path $VALID_DATA_PATH \

CMD=" \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --dataloader-type single \
    --num-workers 0 \
    $DEEPSPEED_ARGS \
    "

echo $CMD


c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# # Bind mask for two threads per core
# BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo "START $SLURM_JOBID: $(date)"

if [ ! -d $wd/cray-deps ] ; then
  rm -rf $wd/cray-deps
  mkdir $wd/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi
    # --cpu-bind=mask_cpu:$BIND_MASK \
    # --cpu-bind=mask_cpu:$BIND_MASK \

srun \
    --label \
    singularity exec \
    -B /opt/cray:/opt/cray \
    -B "$wd/cray-deps":/opt/cray-deps \
    -B "$wd":/workdir \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"