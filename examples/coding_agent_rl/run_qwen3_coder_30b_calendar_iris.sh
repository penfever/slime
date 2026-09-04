#!/usr/bin/env bash
# Five-node Qwen3-Coder calendar RL driver for infra/iris/launcher.py.

set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${LEARNING_RATE:?LEARNING_RATE is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${DAYTONA_API_KEY:?DAYTONA_API_KEY is required}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/app/assets/hf-model}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/app/assets/torch-dist}"
PROMPT_DATA="${PROMPT_DATA:-/app/assets/calendar-v3-slime.jsonl}"
PARTITION_DIR="${PARTITION_DIR:-/tmp/slime-calendar-data}"
TRAIN_PROMPT_DATA="${PARTITION_DIR}/train.jsonl"
EVAL_PROMPT_DATA="${PARTITION_DIR}/eval.jsonl"
SAVE_PATH="${SAVE_PATH:-/app/checkpoints/${RUN_ID}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

for path in \
    "${HF_CHECKPOINT}/config.json" \
    "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" \
    "${PROMPT_DATA}" \
    /app/assets/node-v22.20.0-linux-x64.tar.xz \
    /app/assets/claude-code-2.1.259.tgz; do
    [[ -e "${path}" ]] || { echo "missing required asset: ${path}" >&2; exit 1; }
done

mkdir -p "${PARTITION_DIR}"
mkdir -p "${SAVE_PATH}"
partition_args=(
    --input "${PROMPT_DATA}"
    --train "${TRAIN_PROMPT_DATA}"
    --eval "${EVAL_PROMPT_DATA}"
    --eval-size "${EVAL_SIZE:-128}"
    --seed "${EVAL_SEED:-42}"
)
if [[ -n "${TRAIN_SIZE:-}" ]]; then
    partition_args+=(--train-size "${TRAIN_SIZE}" --train-seed "${TRAIN_SUBSET_SEED:-42}")
fi
python3 examples/coding_agent_rl/partition_jsonl.py "${partition_args[@]}"

export MODEL_ARGS_ROTARY_BASE=10000000
export PYTHONPATH="/root/Megatron-LM:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0

export SWE_AGENT=claude_code
export SWE_TRAIN_PROTOCOL=scaleswe
export SWE_USE_SAMPLE_PROMPT=1
export SWE_CC_PROMPT="${SWE_CC_PROMPT:-Read PROBLEM_STATEMENT.md in full and complete its final unresolved user request. The file is a transcript: the last user message intentionally has no following assistant answer. Apply that request and every earlier constraint instead of copying an earlier assistant calendar. Follow the output-file contract exactly. Do not inspect hidden evaluator state. When finished, print a one-line summary and exit.}"
export SLIME_AGENT_SANDBOX_BACKEND=daytona
export SLIME_AGENT_NODE_TARBALL=/app/assets/node-v22.20.0-linux-x64.tar.xz
export SLIME_AGENT_CC_TARBALL=/app/assets/claude-code-2.1.259.tgz
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-900}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"
export SLIME_AGENT_SANDBOX_LIFETIME_SEC="${SLIME_AGENT_SANDBOX_LIFETIME_SEC:-3600}"

source scripts/models/qwen3-30B-A3B.sh

exec python3 -u train.py \
    --actor-num-nodes 2 \
    --actor-num-gpus-per-node 8 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_MODEL_PATH}" \
    --save "${SAVE_PATH}" \
    --save-interval "${SAVE_INTERVAL}" \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --prompt-data "${TRAIN_PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --num-rollout 400 \
    --rollout-shuffle \
    --rollout-seed "${ROLLOUT_SEED:-42}" \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --rollout-max-context-len 32768 \
    --rollout-max-response-len 4096 \
    --rollout-temperature 1.0 \
    --eval-interval "${EVAL_INTERVAL:-2}" \
    --eval-prompt-data calendar-v3-heldout "${EVAL_PROMPT_DATA}" \
    --n-samples-per-eval-prompt 1 \
    --eval-temperature 0.0 \
    --eval-max-response-len 4096 \
    --num-steps-per-rollout 1 \
    --global-batch-size 256 \
    --micro-batch-size 1 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 8 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --max-tokens-per-gpu 4096 \
    --log-probs-chunk-size 1024 \
    --use-dynamic-batch-size \
    --advantage-estimator grpo \
    --kl-loss-coef 0 \
    --kl-coef 0 \
    --entropy-coef 0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr "${LEARNING_RATE}" \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --clip-grad 0.9 \
    --optimizer-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer \
    --rollout-num-gpus 24 \
    --rollout-num-gpus-per-engine 4 \
    --sglang-server-concurrency 5 \
    --sglang-mem-fraction-static 0.75 \
    --sglang-tool-call-parser qwen3_coder \
    --sglang-reasoning-parser qwen3 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --use-wandb \
    --wandb-project qwen3coder-calendar-slime \
    --wandb-group "${RUN_ID}" \
    --disable-wandb-random-suffix \
    --wandb-always-use-train-step \
    --log-passrate \
    --log-multi-turn
