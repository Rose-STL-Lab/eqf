# Inference and reproducing results

All public inference commands use `ckpt_map=default`, using the checkpoint locations in `configurations/ckpt_map/default.yaml`. The commands below run one seed over 256 videos from the validation dataset.

## Canonical EqF inference

### Minecraft

Run base open loop EqF with GD or closed loop NAG for 250 denoising steps on 300-frame videos:

```bash
python -m main \
  shortcode=exp/minecraft/eqf/infer \
  +name=minecraft_eqf_infer

python -m main \
  shortcode=exp/minecraft/eqf/infer_nag \
  +name=minecraft_nlp_eqf_nag_infer
```

The recipe uses a 50-frame model window, 25 context frames, a stride of 1, and the truncated c function schedule.

### RE10K

Run pose-conditioned open loop EqF GD or closed loop EqF NAG for 50 steps:

```bash
python -m main \
  shortcode=exp/re10k/eqf/infer \
  +name=re10k_eqf_infer

python -m main \
  shortcode=exp/re10k/eqf/infer_nag \
  +name=re10k_nlp_eqf_nag_infer
```

Each 189-frame clip has 37 initial context frames and 152 generated frames.
Generation proceeds in 12-frame chunks through the Wan 2.1 temporal VAE.

### DROID

Run the 50 step budget adaptive EqF NAG on Droid:

```bash
python -m main \
  shortcode=exp/droid/eqf/infer_adaptive \
  ckpt_map=default \
  +name=droid_eqf_budget_adaptive
```

This uses Wan 2.2 to generate 36 frames from 13 context frames. NAG uses
`mu=0.1`.

## Primary table launchers

Running a launcher without a selector evaluates all of its methods with seed 42
and 256 videos. Always inspect the generated commands first:

```bash
bash scripts/minecraft_inference.sh --list
bash scripts/minecraft_inference.sh --dry-run
```

### Minecraft primary table

```bash
bash scripts/minecraft_inference.sh
```

The reindexed methods are:

1. Closed loop EqF + NAG with `mu=0.3`, using the readout-predicted solver index.
2. EqM NAG with `mu=0.1`, using the prescribed schedule.
3. Open loop EqF + GD.
4. FM + Euler
5. Diffusion (DFoT) + DDIM.

EqF, EqM, and FM use the truncated c-function schedule. DFoT is a discrete
DDIM sampler, so the continuous c-function schedule does not apply. The default
budget is 250 steps and can be changed with `--num-sampling-steps`.

### RE10K primary table

```bash
bash scripts/re10k_inference.sh
```

The reindexed methods are:

1. Budget adaptive closed loop EqF NAG with `mu = 0.1`.
2. Fixed closed loop EqF NAG with `mu = 0.1`.
3. Fixed open loop EqF GD.
4. FM Euler.

All EqF methods use the truncated c function schedule. FM uses its identity
schedule. The default budget is 50 steps.

## Budget adaptive results

The three budget adaptive launchers share the same method indices:

1. Fixed closed loop EqF NAG with `mu = 0.1`.
2. Budget adaptive closed loop EqF NAG with `mu = 0.1`.
3. Fixed FM Euler.

By default each method runs at 10, 20, 30, 40, and 50 steps, for 15 jobs per
dataset. 

```bash
bash scripts/adaptive_minecraft_inference.sh
bash scripts/adaptive_re10k_inference.sh
bash scripts/adaptive_droid_inference.sh
```

The dataset-specific rollout geometries for adaptive experiments are:

- Minecraft: 300 frames, rolling inference, 25 context frames, stride 5.
- RE10K: 189 frames, rolling inference, 29 context frames, stride 4.
- DROID: one 49 frame clip, 13 context frames, 36 generated frames.

Minecraft FM uses the truncated c-function schedule, RE10K FM uses identity,
and DROID FM uses its default unwarped schedule.

## Selecting runs and budgets

All five launchers support the same core interface:

```bash
# Show method indices
bash scripts/adaptive_minecraft_inference.sh --list

# Run methods 1 and 3 at two budgets
bash scripts/adaptive_minecraft_inference.sh \
  --run-indices 1,3 \
  --num-sampling-steps 10,50

# Preview commands with a smaller evaluation subset
bash scripts/re10k_inference.sh \
  --dry-run \
  --run-indices 1,2 \
  --num-videos 16
```

Common options include `--seed`, `--ckpt-map`, `--batch-size`, `--devices`,
`--num-sampling-steps`, `--num-videos`, `--wandb-mode`, repeated `--tag`
arguments, and `--extra` followed by arbitrary Hydra overrides.
