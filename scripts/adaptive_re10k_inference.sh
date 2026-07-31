#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

SCRIPT_NAME="adaptive_re10k_inference.sh"
SCRIPT_DESCRIPTION="Run the RE10K fixed/budget-adaptive compute comparison."
RUN_PREFIX="adaptive_re10k_inference"
BASE_TAG="adaptive_re10k_inference"
SEEDS_CSV="42"
CKPT_MAP="default"
BATCH_SIZE=16
DEVICES=1
NUM_SAMPLING_STEPS_CSV="10,20,30,40,50"
NUM_VIDEOS=256
WANDB_MODE="online"
RAW_DIR_BASE=""

JOB_NAMES=("eqf_nag_mu01" "eqf_budget_adaptive_nag_mu01" "fm_euler")
JOB_SHORTCODES=(
  "exp/re10k/eqf/infer_pose_noiselevel"
  "exp/re10k/eqf/infer_pose_noiselevel"
  "exp/re10k/flowf/infer_pose"
)
JOB_ALGORITHMS=(
  "wan_pose_noiselevelpred_eqf_video"
  "wan_pose_noiselevelpred_eqf_video"
  "wan_t2v_pose_forcing_video"
)
JOB_TAGS=("eqf_nag_mu01" "eqf_budget_adaptive_nag_mu01" "fm_euler")

SHARED_OVERRIDES=(
  dataset.n_frames=189
  dataset.max_frames=189
  dataset.clip_sampling_seed=42
  "experiment.tasks=[validation]"
  experiment.validation.precision=bf16
  experiment.validation.sample_during_training=true
  experiment.validation.dataloader.shuffle=false
  algorithm.gradient_checkpointing_rate=0.0
  algorithm.backbone.forward_window_size=49
  algorithm.tasks.prediction.streaming.schedule=rolling
  algorithm.tasks.prediction.streaming.should_bake_in=true
  algorithm.tasks.prediction.streaming.sliding_context_frames=29
  algorithm.tasks.prediction.streaming.initial_context_frames=29
  algorithm.tasks.prediction.streaming.stride_in_frames=4
  algorithm.denoising.history_guidance_scale=0.0
  "algorithm.logging.metrics=[fvd,fvd_over_time,lpips,mse,psnr,vbench,ssim]"
  algorithm.logging.fvd_over_time.clip_len=25
  algorithm.logging.fvd_over_time.gen_stride=1
  algorithm.logging.fvd_over_time.ref_stride=1
  algorithm.logging.fvd_over_time.reference_mode=time_aligned
  algorithm.logging.log_global_denoising_schedule=true
  algorithm.logging.log_frame_nfe=true
  algorithm.logging.log_global_velocity_schedule=true
)

build_job_overrides() {
  local index="$1"
  local nsteps="$2"
  local -a eqf_base=(
    algorithm.denoising.equilibrium_schedule=constant
    algorithm.denoising.equilibrium_lambda=1.0
  )
  local -a cfunc=(
    algorithm.denoising.inference_schedule.enabled=true
    algorithm.denoising.inference_schedule.family=c_function
    algorithm.denoising.inference_schedule.t_end=0.001
    algorithm.denoising.inference_schedule.c_function.schedule=truncated
    algorithm.denoising.inference_schedule.c_function.truncated_a=0.8
  )
  local -a nlp=(
    "algorithm.logging.inference_step_to_noise_level_thresholds=[0.7,0.5,0.3,0.2,0.1,0.05,0.01]"
    algorithm.logging.log_global_readout_noise_level_schedule=true
  )

  case "$index" in
    1)
      JOB_OVERRIDES=(
        "${eqf_base[@]}"
        "${cfunc[@]}"
        "${nlp[@]}"
        algorithm.tasks.prediction.streaming.mode=fixed
        algorithm.tasks.prediction.streaming.tail_finish.enabled=false
        algorithm.denoising.sampling_algorithm=ngd
        algorithm.denoising.mu=0.1
        algorithm.denoising.inference_schedule.inference_solver_index_source=readout_predicted
      )
      ;;
    2)
      JOB_OVERRIDES=(
        "${eqf_base[@]}"
        "${cfunc[@]}"
        "${nlp[@]}"
        algorithm.tasks.prediction.streaming.mode=adaptive
        algorithm.tasks.prediction.streaming.stop_based_on=readout
        algorithm.tasks.prediction.streaming.emit_eps=0.0
        algorithm.tasks.prediction.streaming.frame_done_eps=0.0
        algorithm.tasks.prediction.streaming.emit_noise_level=1.0
        algorithm.tasks.prediction.streaming.frame_done_noise_level=0.0
        algorithm.tasks.prediction.streaming.max_cycles_without_emit=default
        algorithm.tasks.prediction.streaming.tail_finish.enabled=true
        "algorithm.tasks.prediction.streaming.tail_finish.n_cleanup_steps=${nsteps}"
        algorithm.tasks.prediction.streaming.tail_finish.step_multiplier=1.0
        algorithm.tasks.prediction.streaming.tail_finish.disable_momentum=false
        algorithm.denoising.sampling_algorithm=ngd
        algorithm.denoising.mu=0.1
        algorithm.denoising.inference_schedule.inference_solver_index_source=readout_predicted
      )
      ;;
    3)
      JOB_OVERRIDES=(
        algorithm.tasks.prediction.streaming.mode=fixed
        algorithm.tasks.prediction.streaming.tail_finish.enabled=false
        algorithm.denoising.sampling_algorithm=euler
        algorithm.denoising.mu=0.0
        algorithm.denoising.inference_schedule.enabled=true
        algorithm.denoising.inference_schedule.family=identity
        algorithm.denoising.inference_schedule.inference_solver_index_source=schedule
      )
      ;;
    *)
      echo "Unknown adaptive RE10K method index ${index}" >&2
      exit 3
      ;;
  esac
}

source "${SCRIPT_DIR}/inference_sweep_common.sh"
run_inference_sweep "$@"
