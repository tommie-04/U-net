"""Compare saved U-Net epochs on a fixed set of original Carvana images.

This is a descriptive sample comparison, not a recovered held-out evaluation.
The upstream architecture, preprocessing, and predict_img function are reused.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-zip', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--upstream-dir', type=Path, default=Path('Pytorch-UNet'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_dir.resolve()))
    from unet import UNet
    from predict import predict_img

    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_id = 'a1038b207299_14'
    with ZipFile(args.image_zip) as archive:
        entries = {Path(n).stem: n for n in archive.namelist() if n.endswith('.jpg')}
        cars = sorted({n.rsplit('_', 1)[0] for n in entries})
        chosen = [cars[i] + '_01' for i in np.linspace(0, len(cars)-1, 8, dtype=int)]
        sample_ids = list(dict.fromkeys([original_id] + chosen))
        assert all(n in entries for n in sample_ids)
        results = []
        for epoch in [3, 5]:
            state = torch.load(args.checkpoint_dir / f'checkpoint_epoch{epoch}.pth', map_location='cpu')
            values = state.pop('mask_values')
            assert values == [0, 1]
            model = UNet(n_channels=3, n_classes=2, bilinear=False).eval()
            model.load_state_dict(state, strict=True)
            del state
            for sample_id in sample_ids:
                with Image.open(io.BytesIO(archive.read(entries[sample_id]))) as raw:
                    image = raw.convert('RGB')
                with Image.open(args.mask_dir / f'{sample_id}_mask.gif') as raw:
                    truth = np.array(raw) > 0
                pred = predict_img(model, image, torch.device('cpu'), scale_factor=0.5) > 0
                assert pred.shape == truth.shape
                denominator = int(pred.sum()) + int(truth.sum())
                dice = 2 * int(np.logical_and(pred, truth).sum()) / denominator if denominator else 1.0
                Image.fromarray(pred).save(args.output_dir / f'{sample_id}_epoch{epoch}.png')
                results.append({'sample_id': sample_id, 'epoch': epoch, 'dice_original_resolution': dice})
                print(f'epoch={epoch} sample={sample_id} Dice={dice:.6f}', flush=True)
            del model
        with (args.output_dir / 'sample_scores.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(results[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(results)
        lookup = {(r['sample_id'], r['epoch']): r['dice_original_resolution'] for r in results}
        # Select a good qualitative example by epoch-3 Dice; disclose selection.
        selected = max(sample_ids, key=lambda n: lookup[(n, 3)])
        for sample_id, filename in [(original_id, 'original_sample_epoch_comparison.png'), (selected, 'selected_example.png')]:
            with Image.open(io.BytesIO(archive.read(entries[sample_id]))) as raw:
                rgb = np.array(raw.convert('RGB'))
            with Image.open(args.mask_dir / f'{sample_id}_mask.gif') as raw:
                truth = np.array(raw) > 0
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            axes[0].imshow(rgb)
            axes[0].set_title('Original')
            axes[1].imshow(truth, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Ground truth')
            for ax, epoch in zip(axes[2:], [3, 5]):
                with Image.open(args.output_dir / f'{sample_id}_epoch{epoch}.png') as raw:
                    pred = np.array(raw) > 0
                ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Epoch {epoch} | Dice {lookup[(sample_id, epoch)]:.4f}')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f'{sample_id}')
            fig.tight_layout()
            fig.savefig(args.output_dir / filename, dpi=170)
            plt.close(fig)
        # Keep the original display image for the README, avoiding sample switching.
        with Image.open(io.BytesIO(archive.read(entries[original_id]))) as raw:
            rgb = np.array(raw.convert('RGB'))
        with Image.open(args.mask_dir / f'{original_id}_mask.gif') as raw:
            truth = np.array(raw) > 0
        with Image.open(args.output_dir / f'{original_id}_epoch3.png') as raw:
            pred = np.array(raw) > 0
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title('Original')
        for ax, data, title in [
            (axes[1], truth, 'Ground truth'),
            (axes[2], pred, f'Epoch 3 prediction | Dice {lookup[(original_id, 3)]:.4f}'),
        ]:
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f'{original_id}')
        fig.tight_layout()
        fig.savefig(args.output_dir / 'prediction_comparison_epoch3.png', dpi=180)
        plt.close(fig)
        checkpoint_hashes = {}
        for epoch in [3, 5]:
            path = args.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
            with path.open('rb') as source:
                digest = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    digest.update(chunk)
                checkpoint_hashes[str(epoch)] = digest.hexdigest()
        summary = {
            'inference_environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'scale': 0.5},
            'epochs_compared': [3, 5],
            'checkpoint_sha256': checkpoint_hashes,
            'sample_ids': sample_ids,
            'sampling_rule': 'Original displayed sample plus angle 01 of eight evenly spaced car IDs in lexicographic order.',
            'showcase_selection': 'Auxiliary selected_example.png only: highest epoch-3 Dice among this fixed sample set; not the README image.',
            'readme_display_sample': original_id,
            'readme_selection_reason': 'Retained the original display image; checkpoint selected by logged epoch-end validation Dice.',
            'selected_sample': selected,
            'original_sample_dice': {str(e): lookup[(original_id, e)] for e in [3,5]},
            'selected_sample_dice': {str(e): lookup[(selected, e)] for e in [3,5]},
            'mean_sample_dice': {str(e): float(np.mean([lookup[(n, e)] for n in sample_ids])) for e in [3,5]},
            'limitations': 'Same acquired training-image pool; actual split membership unknown. Sample Dice uses original-resolution predictions, unlike logged resized-resolution validation. Local inference package versions differ from cloud training. These sample scores are not an independent test score or a replacement for logged validation.'
        }
        (args.output_dir / 'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
