# Epoch-3 and Epoch-5 Comparison

Epoch 3 had the highest epoch-end validation Dice among the saved models:
0.981803, compared with 0.968135 for epoch 5. I used it for the README example
and downloadable checkpoint.

## Nine-image comparison

The sample set contains the original display image plus angle `01` of eight
car IDs spaced evenly through the sorted car-ID list. The scores below use
predicted masks at the original image resolution.

| Image ID | Epoch 3 Dice | Epoch 5 Dice |
| --- | --- | --- |
| `a1038b207299_14` | 0.990765 | 0.933728 |
| `00087a6bd4dc_01` | 0.994340 | 0.992397 |
| `26ccab021981_01` | 0.977790 | 0.982833 |
| `40851e617b93_01` | 0.981918 | 0.995367 |
| `683ddec95b82_01` | 0.981169 | 0.961999 |
| `880fec285c06_01` | 0.985612 | 0.995196 |
| `bb2f95262773_01` | 0.992804 | 0.993589 |
| `dc995bf5296a_01` | 0.992324 | 0.994994 |
| `fff9b3a5373f_01` | 0.992583 | 0.993421 |
| **Mean of these nine images** | **0.987700** | **0.982614** |

The [side-by-side figure](figures/original_sample_epoch_comparison.png) shows
both predictions for the original image. The [README](../README.md#prediction-comparison)
shows the selected epoch-3 result.

## Repeat the comparison

Run from the outer project folder with the official checkout available. The
script reads images from `train_hq.zip`, extracted GIF masks, and saved weights.
Choose a new output directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-inference.txt
python scripts/compare_checkpoints.py --image-zip /path/to/train_hq.zip --mask-dir /path/to/train_masks --checkpoint-dir /path/to/checkpoints --output-dir artifacts/new-checkpoint-comparison
```

The [script](../scripts/compare_checkpoints.py) saves predicted masks, figures,
and a CSV of per-image Dice. The [recorded scores](../records/checkpoint-comparison/sample_scores.csv)
are available alongside the [comparison summary](../records/checkpoint-comparison/comparison_summary.json).
