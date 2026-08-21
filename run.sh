#!/usr/bin/env bash
#
# Every experiment in the paper, for every model.
#
#   bash run.sh                                    everything: both attributes, six models
#   bash run.sh --model Qwen/Qwen2.5-3B-Instruct   one model
#   bash run.sh --attribute moral                  one attribute
#   bash run.sh --figures-only                     rebuild outputs/ from results/, no GPU
#
# Every stage is resume-safe: it skips items already on disk. Runs one model at a time.
#
set -euo pipefail

MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "LiquidAI/LFM2-2.6B-Exp"
  "HuggingFaceH4/zephyr-7b-beta"
  "HuggingFaceH4/zephyr-7b-alpha"
)

N_EVAL=500
N_SWEEP=200
N_TRAIN=200
STRIDE=1

TOX_TAG=steering_rtp
TOX_ROUNDS=5
TOX_MAX_NEW=200
TOX_VARIANTS="strong_detox weak_detox strong_tox weak_tox neutral_concise"
TOX_ALPHAS="0.5 1.0 2.0"

MORAL_TAG=steering_moral
MORAL_ROUNDS=2
MORAL_MAX_NEW=64
MORAL_VARIANTS="moralize_strong moralize_weak immoralize_strong immoralize_weak moral_neutral"
MORAL_ALPHAS="0.5 1.0 2.0"

ATTRIBUTE=both
FIGURES_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model)        MODELS=("$2"); shift 2 ;;
    --attribute)    ATTRIBUTE="$2"; shift 2 ;;
    --figures-only) FIGURES_ONLY=1; shift ;;
    -h|--help)      awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *)              echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${ATTRIBUTE}" in toxicity|moral|both) ;; *)
  echo "--attribute must be toxicity, moral or both" >&2; exit 2 ;;
esac

if [ "${FIGURES_ONLY}" -eq 0 ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Create a token at https://huggingface.co/settings/tokens," >&2
  echo "accept the licence for any gated model, then:  export HF_TOKEN=..." >&2
  exit 1
fi
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

step () { echo; echo "===== $*"; }
py   () { python -m "$@"; }

if [ "${FIGURES_ONLY}" -eq 1 ]; then
  py experiments.make_figures
  exit 0
fi

step "data splits and contrastive pairs"
SOURCES=""
[ "${ATTRIBUTE}" != "moral" ] && { py experiments.pipeline prepare_toxicity; SOURCES="rtp jigsaw"; }
[ "${ATTRIBUTE}" != "toxicity" ] && { py experiments.pipeline prepare_moral; SOURCES="${SOURCES} moral ethics"; }
py experiments.pipeline contrastive_pairs --sources ${SOURCES}

if [ "${ATTRIBUTE}" != "toxicity" ]; then
  step "the moral scorer (once, shared by every model)"
  if [ -d results/_scorers/moral_roberta ]; then
    py experiments.score train_moral_scorer --eval_only
  else
    py experiments.score train_moral_scorer
  fi
fi

py experiments.pipeline check_models "${MODELS[@]}"

run_toxicity () {
  local M="$1"

  step "[toxicity] steering vectors -- mu_rtp and mu_jigsaw"
  py experiments.pipeline steering_vectors --model "$M" --sources rtp
  py experiments.pipeline steering_vectors --model "$M" --sources jigsaw

  step "[toxicity] self-correction dialog"
  py experiments.pipeline self_correction --model "$M" --num_data "${N_EVAL}" \
     --num_rounds "${TOX_ROUNDS}" --variants ${TOX_VARIANTS}
  for v in strong_detox strong_tox; do
    py experiments.pipeline self_correction --model "$M" --split train --num_data "${N_TRAIN}" \
       --num_rounds 2 --variants "$v"
  done

  step "[toxicity] residual norms -- the scale every alpha is a fraction of"
  py experiments.measure residual_norms --model "$M"

  step "[toxicity] inject mu (Fig 1, top row)"
  py experiments.inject mu --model "$M" --steering_tag "${TOX_TAG}" \
     --alphas ${TOX_ALPHAS} --layer_stride "${STRIDE}" --limit "${N_SWEEP}"

  step "[toxicity] inject ell_bar (Fig 1, bottom row -- the paper's causal claim)"
  py experiments.inject shift --model "$M" --steering_tag "${TOX_TAG}" \
     --variant strong_detox --alphas ${TOX_ALPHAS} --layer_stride "${STRIDE}" \
     --limit "${N_SWEEP}" --n_train "${N_TRAIN}"
  py experiments.inject shift --model "$M" --steering_tag "${TOX_TAG}" \
     --variant strong_tox --alphas ${TOX_ALPHAS} --layer_stride "${STRIDE}" \
     --limit "${N_SWEEP}" --n_train "${N_TRAIN}"

  step "[toxicity] alignment, and the Jigsaw transfer check"
  py experiments.measure alignment --model "$M" --steering_tag "${TOX_TAG}" \
     --limit "${N_EVAL}" --num_rounds "${TOX_ROUNDS}" --variants ${TOX_VARIANTS}
  py experiments.measure alignment --model "$M" --steering_tag steering_jigsaw \
     --limit "${N_EVAL}" --num_rounds "${TOX_ROUNDS}" --variants ${TOX_VARIANTS}
  py experiments.measure mu_transfer --model "$M" --attribute toxicity

  step "[toxicity] quality metrics for the by-round table"
  py experiments.score generations --model "$M" --ingest isc \
     --variants ${TOX_VARIANTS} --metrics ppl --limit "${N_EVAL}"
  py experiments.score generations --model "$M" --ingest isc \
     --variants ${TOX_VARIANTS} --metrics judge --limit "${N_EVAL}"
}

run_moral () {
  local M="$1"

  step "[moral] steering vectors -- mu_moral and mu_ethics"
  py experiments.pipeline steering_vectors --model "$M" --sources moral
  py experiments.pipeline steering_vectors --model "$M" --sources ethics

  step "[moral] self-correction dialog"
  py experiments.pipeline self_correction --model "$M" --num_data "${N_EVAL}" \
     --num_rounds "${MORAL_ROUNDS}" --variants ${MORAL_VARIANTS}
  for v in moralize_strong immoralize_strong; do
    py experiments.pipeline self_correction --model "$M" --split train --num_data "${N_TRAIN}" \
       --num_rounds "${MORAL_ROUNDS}" --variants "$v"
  done

  step "[moral] residual norms under the MORAL prefill"
  py experiments.measure residual_norms --model "$M" --isc_variant moralize_strong \
     --steering_tag "${MORAL_TAG}" --source moral --n_train "${N_TRAIN}"

  step "[moral] inject mu_moral (Fig 1, top row)"
  py experiments.inject mu --model "$M" --steering_tag "${MORAL_TAG}" \
     --alphas ${MORAL_ALPHAS} --layer_stride "${STRIDE}" --limit "${N_SWEEP}" \
     --max_new_tokens "${MORAL_MAX_NEW}"

  step "[moral] inject ell_bar_moral (Fig 1, bottom row)"
  for v in moralize_strong immoralize_strong; do
    py experiments.inject shift --model "$M" --steering_tag "${MORAL_TAG}" \
       --variant "$v" --alphas ${MORAL_ALPHAS} --layer_stride "${STRIDE}" \
       --limit "${N_SWEEP}" --n_train "${N_TRAIN}" --max_new_tokens "${MORAL_MAX_NEW}"
  done

  step "[moral] alignment, and the ETHICS transfer check"
  py experiments.measure alignment --model "$M" --steering_tag "${MORAL_TAG}" \
     --limit "${N_EVAL}" --num_rounds "${MORAL_ROUNDS}" --n_null 100 --variants ${MORAL_VARIANTS}
  py experiments.measure alignment --model "$M" --steering_tag steering_ethics \
     --limit "${N_EVAL}" --num_rounds "${MORAL_ROUNDS}" --n_null 100 --variants ${MORAL_VARIANTS}
  py experiments.measure mu_transfer --model "$M" --attribute moral
}

for M in "${MODELS[@]}"; do
  echo
  echo "##################################################################"
  echo "##  $(basename "$M")  ($(date +%H:%M:%S))"
  echo "##################################################################"
  [ "${ATTRIBUTE}" != "moral" ]    && run_toxicity "$M"
  [ "${ATTRIBUTE}" != "toxicity" ] && run_moral "$M"
done

step "figures and tables"
py experiments.make_figures

echo
echo "done. figures and tables -> outputs/"
