# conda activate gcl
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=3,4
export DEEPSPEED=deepspeed_zero2
export EXPNAME=coin_llava_lora


NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"

NPROC_PER_NODE=$NGPUS xtuner train \
    ./cltuner/configs/$EXPNAME/ScienceQA.py \
    --deepspeed $DEEPSPEED
