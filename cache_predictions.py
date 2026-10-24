"""Cache RF-DETR Nano predictions for reproducible architecture-neutral scoring."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('RF_HOME', str(ROOT / '.cache'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/predictions')
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; partial splits are ineligible for submission.')
    args = parser.parse_args()
    import importlib.metadata
    import numpy as np
    from PIL import Image
    import torch
    from rfdetr import RFDETRNano

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    model = RFDETRNano(pretrain_weights=str(args.weights.resolve()), resolution=640,
                       positional_encoding_size=40, num_classes=18, device=args.device,
                       num_select=300)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    runtime_state = model.model.model.state_dict()
    assert set(runtime_state) == set(checkpoint['model'])
    assert all(torch.equal(runtime_state[key].cpu(), value.cpu()) for key, value in checkpoint['model'].items()), (
        'Inference model parameters differ from checkpoint; do not report its metrics.')
    del checkpoint, runtime_state
    for split in args.splits:
        folder = args.dataset / ('valid' if split == 'val' else 'test')
        annotation_file = folder / '_annotations.coco.json'
        encoded = annotation_file.read_bytes()
        annotation = json.loads(encoded)
        names = [x['name'] for x in sorted(annotation['categories'], key=lambda x: x['id'])]
        assert list(model.class_names) == names, (model.class_names, names)
        images = sorted(annotation['images'], key=lambda x: x['id'])
        if args.limit:
            images = images[:args.limit]
        destination = args.output / f'{split}.jsonl.gz'
        metadata_path = args.output / f'{split}_metadata.json'
        if destination.exists() and metadata_path.exists():
            prior = json.loads(metadata_path.read_text())
            if (prior.get('complete') and prior['weights_sha256'] == weights_sha
                    and prior['annotation_sha256'] == hashlib.sha256(encoded).hexdigest()
                    and prior['images'] == len(images)):
                print(f'{split}: verified existing complete cache', flush=True)
                continue
            raise RuntimeError(f'Refusing to overwrite a different cache: {destination}')
        started = time.perf_counter()
        temporary = destination.with_suffix('.gz.part')
        with gzip.open(temporary, 'wt', encoding='utf-8') as handle:
            for offset in range(0, len(images), args.batch):
                batch_records = images[offset:offset + args.batch]
                batch = []
                for record in batch_records:
                    with Image.open(folder / record['file_name']) as image:
                        image = image.convert('RGB')
                        assert image.size == (record['width'], record['height'])
                        batch.append(image)
                predictions = model.predict(batch, threshold=0.001, include_source_image=False)
                assert len(predictions) == len(batch_records)
                for record, detections in zip(batch_records, predictions):
                    clipped = detections.xyxy.copy()
                    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, record['width'])
                    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, record['height'])
                    # RF-DETR's extra no-object logit is not a dataset category.
                    # Keep degenerate boxes as false positives rather than improving metrics by dropping them.
                    valid = np.flatnonzero((detections.class_id >= 0) & (detections.class_id < 18))
                    order = valid[np.argsort(-detections.confidence[valid], kind='stable')[:300]]
                    classes = detections.class_id[order]
                    assert all(0 <= int(cid) < 18 for cid in classes)
                    output = {'image_id': record['id'], 'xyxy': clipped[order].tolist(),
                              'confidence': detections.confidence[order].tolist(),
                              'class_id': classes.tolist()}
                    handle.write(json.dumps(output, separators=(',', ':')) + '\n')
                completed = min(offset + args.batch, len(images))
                if offset % (args.batch * 10) == 0 or completed == len(images):
                    progress = {'phase': 'predicting', 'split': split, 'images': completed,
                                'total_images': len(images), 'elapsed_seconds': time.perf_counter() - started}
                    (args.output / 'status.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
                    print(json.dumps(progress), flush=True)
        temporary.replace(destination)
        metadata = {'complete': True, 'partial_smoke_test': bool(args.limit), 'split': split,
                    'images': len(images), 'weights_sha256': weights_sha,
                    'annotation_sha256': hashlib.sha256(encoded).hexdigest(),
                    'prediction_cache_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'resolution': 640, 'max_det': 300, 'confidence_floor': 0.001,
                    'nms': None, 'boxes_clipped_to_image': True, 'no_object_slot_excluded': True,
                    'rfdetr': importlib.metadata.version('rfdetr'),
                    'torch': torch.__version__, 'device': args.device,
                    'elapsed_seconds': time.perf_counter() - started,
                    'created_utc': datetime.now(timezone.utc).isoformat()}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
