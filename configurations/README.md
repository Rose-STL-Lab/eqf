# Configuration

The project uses [Hydra](https://hydra.cc/docs/intro/) to compose experiment,
dataset, algorithm, checkpoint, cluster, and user-facing shortcode groups.
Every resolved configuration is saved with its W&B run.

The root defaults are in `config.yaml`. Supported release recipes live under:

- `shortcode/exp/minecraft/`
- `shortcode/exp/re10k/`
- `shortcode/exp/droid/`
- `shortcode/preprocess/preprocess_minecraft.yaml`

Use a shortcode as the base recipe, then override only the values needed for a
run:

```bash
python -m main \
  shortcode=exp/minecraft/eqf/train_eqf \
  ckpt_map=default \
  +name=minecraft_eqf
```

Canonical release inference recipes are:

- `shortcode=exp/minecraft/eqf/infer`
- `shortcode=exp/re10k/eqf/infer`
- `shortcode=exp/droid/eqf/infer_adaptive`

They compose the dataset, algorithm, released checkpoint, evaluation geometry,
and public sampling defaults without additional command-line overrides.

Hydra config groups use slash syntax. For example, select a VAE with
`algorithm/vae=image_vae_minecraft`; `algorithm.vae=<name>` incorrectly
replaces the structured VAE node with a string.

`ckpt_map/default.yaml` defines the portable public checkpoint layout. Add a
custom checkpoint map or cluster config locally when running in another
environment.
