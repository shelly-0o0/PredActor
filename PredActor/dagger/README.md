# Strict DAgger

This package implements learner-origin, isolated-teacher DAgger for PredActor.
It intentionally rejects beta-mixed rollout shards.

For `horizon=20` and `n_obs_steps=4`, one branch archive contains:

- four learner-visited state rows ending at the branch origin;
- three learner action rows used only as context;
- seventeen teacher state/action rows beginning at the exact same origin;
- a command ID and a role (`origin`, `recovery`, `disturbance`, or
  `counterfactual`).

`BranchWindow.assemble()` produces a 20-row window. Its action mask is
`[false, false, false, true, ..., true]`, so learner actions never contribute
to the supervised action or action-difference loss. The collector advances the
live simulator only with the learner and runs the teacher on the object returned
by the simulator adapter's `fork(snapshot)` method. The branch's `observe()`
result is compared with the learner origin before the first teacher action.

Build each immutable shard with:

```bash
python PredActor/scripts/build_strict_dagger_zarr.py \
  --input /path/to/branch_npz_directory \
  --output /path/to/origins.zarr
```

The builder writes a receipt beside the zarr. Training fails closed if a
non-D0 source has no receipt, if the receipt permits learner targets, or if its
role differs from the manifest.

Copy `configs/strict_round1_manifest.example.yaml`, set its environment
variables, then validate it:

```bash
python PredActor/scripts/validate-dagger-manifest.py /path/to/manifest.yaml
```

The default training profile fixes D0 at 60%, uses the pi0 checkpoint's
embedded normalizer, warm-starts only model weights, starts a fresh optimizer,
uses a learning rate of `1.5e-5`, and runs four epochs:

```bash
python PredActor/train.py \
  --cfg g1prdp_dagger_strict.yaml \
  --exp_name strict_round1 \
  --gpu 0
```

Runtime symmetry augmentation is disabled because a matrix reflection cannot
change CLIP's `left` text into `right`. Store mirrored samples with swapped
command IDs in a `counterfactual` shard instead.
