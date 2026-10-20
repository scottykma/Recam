# ReCam demo assets

Two short demo clips and two first-frame stills (one person, one cat) sampled
from the ReCam training corpus. They keep the demo small enough to ship in the
repository.

| File | Content |
|---|---|
| `clip_forest.mp4` | 121 frames, 16 fps, 832x448, person walking an autumn forest road |
| `clip_wetland.mp4` | 121 frames, 16 fps, 832x448, aerial wetland and distant mountain |
| `person_lake.png` | woman walking along a lakeside promenade |
| `cat.png` | gray cat peeking out of a paper bag |
| `demo.parquet` | two `ReCamDataset` rows pointing at the clips |
| `parquets_list.txt` | one-line manifest consumed by the training launcher |

The clips are 832x448 because the shipped recipe trains and infers at that
geometry. Both have 121 frames (`1 + 40 * 3`), which is the context length for
three chunks at 41 frames per chunk. The two stills are used by
`example.recam.infer.prompts.demo`.

The parquet stores repo-relative video paths, so run the launchers from the
repository root (they `cd` there themselves).

These files are short excerpts from the training corpus, included only to
demonstrate the training and inference pipeline.

## Demo training

The recipe in `configs/recam/recam.yaml` uses `n_chunks: 13`; the demo clips are
only long enough for three chunks, so override it for the demo. This one-step
smoke test fits on one GPU by using a reduced-depth model; the full 14B recipe
needs the multi-node setup described in the top-level README:

```bash
MANIFEST=assets/demo/parquets_list.txt GPUS_PER_NODE=1 \
  bash scripts/recam/run_train.sh \
  --data.n-chunks 3 \
  --data.task-probs 0.2 0.4 0.4 \
  --num-steps 1 \
  --dataloader.num-workers 0 \
  --parallel.no-hsdp \
  --model.transformer.num-layers 2
```

It exercises the real dataset, window sampler, VAE/text encoders and one
forward/backward step. The WAN VAE and T5 weights still have to be placed under
`models/` (or passed through `--model.encoder.vae.path` /
`--model.encoder.text-encoder.path`); model weights are not stored in this
repository. Leave `checkpoint.pretrained_path` empty for a random-init smoke
run, or point it at a ReCam checkpoint. The two-row demo parquet is intended for
this single-GPU smoke test; the full 14B recipe needs the multi-node setup and a real
manifest described in the top-level README.

## Demo inference

Once a ReCam checkpoint exists, the default prompt source has two cases:

```bash
CKPT_PATH=weights/recam-14b N_CHUNKS=3 GPUS=0 NPROC_PER_NODE=1 \
  bash scripts/recam/run_infer.sh
```

Use `CASES=demo_person_lake` or `CASES=demo_cat` to run one. `N_CHUNKS=3`
matches the demo clips' 121-frame context. To continue `clip_forest.mp4` instead of starting from a
still:

```bash
CKPT_PATH=weights/recam-14b N_CHUNKS=3 GPUS=0 NPROC_PER_NODE=1 \
PROMPT_SOURCE= \
  bash scripts/recam/run_infer.sh \
  --prompt "The camera slowly follows a person in a red jacket walking along an autumn forest road." \
  --context-video assets/demo/clip_forest.mp4
```

`CKPT_PATH` accepts the checkpoint directory, its `model/` or `ema/` directory,
or a single `.pt`/`.safetensors` file. The checkpoint is not shipped; download
the released weights or point `CKPT_PATH` at a run produced by the training
launcher (`EXP_ID=recam CKPT_ID=step_0002000` also works).
