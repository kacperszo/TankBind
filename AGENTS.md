# TankBind fork

Read CLAUDE.md for architecture, reproduction evidence, and environment constraints.
Make model changes in a separate checkout, not the harness submodule directory.
Run model code through the harness PodmanRunner, offline and with read-only inputs.

## Hard-won facts

- Crystal coordinates supply training distance targets. RDKit coordinates do not
  define a valid native distance target and can reorder atoms through SMILES.
- Harness validation selects affinity checkpoints on RDKit inputs, matching
  embed_complexes.py, including the three-decimal crystal pocket centre and
  conformer seed 0. This is a harness protocol choice, not a claim to reproduce
  the authors' crystal validation split.
- Validation uses batch size 1 to match the inference CLI. The training seed
  does not implicitly change the validation conformer seed.
- Cached graphs depend on labels, source structures, conformer policy and seed.
  A split filename alone is insufficient. v2 uses a separate content namespace;
  never overwrite older caches in place.
- Native pockets remain privileged information; this is not p2rank/decoy training.
- torchdrug can create relative compilation locks. Use a writable container cwd.
- Keep extension fallbacks and bond-stereo trimming; see CLAUDE.md for evidence.
