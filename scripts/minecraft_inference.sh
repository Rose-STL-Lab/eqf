#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

SCRIPT_NAME="minecraft_inference.sh"
SCRIPT_DESCRIPTION="Reproduce the five Minecraft primary-table inference methods."
RUN_PREFIX="minecraft_inference"
BASE_TAG="minecraft_inference"
SEEDS_CSV="42"
CKPT_MAP="default"
BATCH_SIZE=16
DEVICES=1
NUM_SAMPLING_STEPS_CSV="250"
NUM_VIDEOS=256
WANDB_MODE="online"
RAW_DIR_BASE=""

JOB_NAMES=(
  "eqf_nag_mu03"
  "eqm_nag_mu01"
  "eqf_gd_open_loop"
  "fm_euler"
  "dfot_ddim"
)
JOB_SHORTCODES=(
  "exp/minecraft/eqf/infer_noiselevel"
  "exp/minecraft/eqm/infer_trunc_lambda4"
  "exp/minecraft/eqf/infer_noiselevel"
  "exp/minecraft/flowf/infer"
  "exp/minecraft/dfot/infer"
)
JOB_ALGORITHMS=(
  "noiselevelpred_eqf_video"
  "eqf_video"
  "noiselevelpred_eqf_video"
  "flowf_video"
  "dfot_video"
)
JOB_TAGS=(
  "eqf_nag_mu03"
  "eqm_nag_mu01"
  "eqf_gd_open_loop"
  "fm_euler"
  "dfot_ddim"
)

SHARED_OVERRIDES=(
  dataset.n_frames=300
  dataset.max_frames=300
  dataset.clip_sampling_seed=42
  "experiment.tasks=[validation]"
  experiment.validation.precision=16
  experiment.validation.sample_during_training=true
  experiment.validation.dataloader.shuffle=false
  algorithm.backbone.forward_window_size=50
  algorithm.tasks.prediction.streaming.mode=fixed
  algorithm.tasks.prediction.streaming.sliding_context_frames=25
  algorithm.tasks.prediction.streaming.initial_context_frames=25
  algorithm.tasks.prediction.streaming.stride_in_frames=1
  algorithm.logging.log_global_denoising_schedule=true
  algorithm.logging.log_frame_nfe=true
  algorithm.logging.log_global_velocity_schedule=true
)

build_job_overrides() {
  local index="$1"
  local _nsteps="$2"
  local -a cfunc=(
    algorithm.denoising.inference_schedule.enabled=true
    algorithm.denoising.inference_schedule.family=c_function
    algorithm.denoising.inference_schedule.t_end=0.001
    algorithm.denoising.inference_schedule.c_function.schedule=truncated
    algorithm.denoising.inference_schedule.c_function.truncated_a=0.8
  )
  local -a nlp_logging=(
    "algorithm.logging.inference_step_to_noise_level_thresholds=[0.5,0.3,0.1,0.05,0.01]"
    algorithm.logging.log_global_readout_noise_level_schedule=true
  )

  case "$index" in
    1)
      JOB_OVERRIDES=(
        "${cfunc[@]}"
        "${nlp_logging[@]}"
        algorithm.tasks.prediction.streaming.tail_finish.enabled=false
        algorithm.denoising.sampling_algorithm=ngd
        algorithm.denoising.mu=0.3
        algorithm.denoising.inference_schedule.inference_solver_index_source=readout_predicted
      )
      ;;
    2)
      JOB_OVERRIDES=(
        "${cfunc[@]}"
        algorithm.denoising.sampling_algorithm=ngd
        algorithm.denoising.mu=0.1
        algorithm.denoising.inference_schedule.inference_solver_index_source=schedule
      )
      ;;
    3)
      # The readout-enabled checkpoint is loaded, but the solver remains open
      # loop by indexing from the prescribed schedule.
      JOB_OVERRIDES=(
        "${cfunc[@]}"
        "${nlp_logging[@]}"
        algorithm.tasks.prediction.streaming.tail_finish.enabled=false
        algorithm.denoising.sampling_algorithm=gd
        algorithm.denoising.mu=0.0
        algorithm.denoising.inference_schedule.inference_solver_index_source=schedule
      )
      ;;
    4)
      JOB_OVERRIDES=(
        "${cfunc[@]}"
        algorithm.denoising.sampling_algorithm=euler
        algorithm.denoising.mu=0.0
        algorithm.denoising.inference_schedule.inference_solver_index_source=schedule
      )
      ;;
    5)
      # DFoT is discrete DDIM; the continuous c-function schedule does not
      # apply to this baseline.
      JOB_OVERRIDES=(
        algorithm.denoising.sampling_algorithm=ddim
      )
      ;;
    *)
      echo "Unknown Minecraft method index ${index}" >&2
      exit 3
      ;;
  esac
}

source "${SCRIPT_DIR}/inference_sweep_common.sh"
run_inference_sweep "$@"
