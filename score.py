"""Score cached predictions using the same AP/F1 implementation as the YOLO baseline."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def load_split(dataset, cache, split, names):
    import numpy as np
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    folder = dataset / ('valid' if split == 'val' else 'test')
    encoded = (folder / '_annotations.coco.json').read_bytes()
    annotation = json.loads(encoded)
    metadata = json.loads((cache / f'{split}_metadata.json').read_text())
    assert metadata['complete'] and not metadata['partial_smoke_test']
    assert metadata['annotation_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert metadata['prediction_cache_sha256'] == hashlib.sha256((cache / f'{split}.jsonl.gz').read_bytes()).hexdigest()
    assert metadata['max_det'] == 300 and metadata['confidence_floor'] == 0.001
    assert [x['name'] for x in sorted(annotation['categories'], key=lambda x: x['id'])] == list(names.values())
    labels = {image['id']: [] for image in annotation['images']}
    for item in annotation['annotations']:
        assert item.get('iscrowd', 0) == 0
        x, y, w, h = item['bbox']
        labels[item['image_id']].append((item['category_id'], [x, y, x + w, y + h]))
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    stats = []; records = []; seen = set()
    with gzip.open(cache / f'{split}.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            prediction = json.loads(line)
            image_id = prediction['image_id']
            assert image_id in labels and image_id not in seen
            seen.add(image_id)
            boxes = torch.tensor(prediction['xyxy'], dtype=torch.float32).reshape(-1, 4)
            confidence = torch.tensor(prediction['confidence'], dtype=torch.float32)
            classes = torch.tensor(prediction['class_id'], dtype=torch.int64)
            assert len(boxes) == len(confidence) == len(classes) and len(boxes) <= 300
            assert torch.isfinite(boxes).all() and torch.isfinite(confidence).all()
            assert ((classes >= 0) & (classes < 18)).all()
            order = torch.argsort(confidence, descending=True, stable=True)
            preds = {'bboxes': boxes[order], 'conf': confidence[order], 'cls': classes[order]}
            ground_truth = labels[image_id]
            targets = {'cls': torch.tensor([x[0] for x in ground_truth], dtype=torch.int64),
                       'bboxes': torch.tensor([x[1] for x in ground_truth], dtype=torch.float32).reshape(-1, 4)}
            correct = torch.zeros((len(boxes), 10), dtype=torch.bool)
            if len(boxes) and ground_truth:
                correct = BaseValidator.match_predictions(matcher, preds['cls'], targets['cls'],
                                                          box_iou(targets['bboxes'], preds['bboxes']))
            stats.append((correct.numpy(), preds['conf'].numpy(), preds['cls'].numpy(), targets['cls'].numpy()))
            records.append((preds, targets))
    assert seen == set(labels) and len(seen) == metadata['images']
    return tuple(np.concatenate([s[i] for s in stats], axis=0) for i in range(4)), records, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'reports/predictions')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/evaluation')
    args = parser.parse_args()
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix
    assert ultralytics.__version__ == '8.4.182', 'Match baseline metric implementation exactly.'
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'metrics.json').exists():
        raise RuntimeError('Final evaluation already exists; use a new output directory.')
    categories = json.loads((args.dataset / 'valid/_annotations.coco.json').read_text())['categories']
    names = {x['id']: x['name'] for x in sorted(categories, key=lambda x: x['id'])}
    rows = []; selected = None; weights_sha = None
    summary = {'model': 'RF-DETR Nano', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'metric_implementation': f'Ultralytics {ultralytics.__version__}',
               'resolution': 640, 'ap_confidence_floor': 0.001, 'max_det': 300, 'nms': None,
               'checkpoint_selection': 'Maximum validation EMA COCO mAP@50-95.',
               'threshold_selection': 'Maximum unsmoothed mean per-class F1 on validation; same grid as baseline.',
               'f1_definition': 'Mean of per-class harmonic P/R at IoU 0.50, using validation-selected confidence.',
               'test_note': 'Internal benchmark also used in earlier experiments; not an independent field evaluation.'}
    for split in ['val', 'test']:
        stats, records, metadata = load_split(args.dataset, args.cache, split, names)
        expected = (2046, 4758) if split == 'val' else (1059, 2669)
        assert (len(records), len(stats[3])) == expected
        if weights_sha is None:
            weights_sha = metadata['weights_sha256']
        assert weights_sha == metadata['weights_sha256']
        out = args.output / split
        out.mkdir(exist_ok=True)
        result = ap_per_class(*stats, plot=True, save_dir=out, names=names)
        _, _, _, _, _, ap, class_ids, p_curve, r_curve, f1_curve, px, _ = result
        assert class_ids.tolist() == list(range(18))
        if selected is None:
            eligible = np.flatnonzero(px >= 0.001)
            index = int(eligible[np.argmax(f1_curve.mean(0)[eligible])])
            selected = float(px[index])
            summary['selected_confidence'] = selected
        values = []
        for index, cid in enumerate(class_ids):
            precision = float(np.interp(selected, px, p_curve[index]))
            recall = float(np.interp(selected, px, r_curve[index]))
            f1 = 2 * precision * recall / (precision + recall + 1e-16)
            values.append((precision, recall, f1))
            rows.append({'split': split, 'class_id': int(cid), 'class_name': names[int(cid)],
                         'gt_instances': int((stats[3] == cid).sum()),
                         'images_with_class': sum(bool((target['cls'] == int(cid)).any()) for _, target in records),
                         'confidence': selected, 'precision_iou50': precision, 'recall_iou50': recall,
                         'f1_iou50': f1, 'ap50': float(ap[index, 0]), 'ap50_95': float(ap[index].mean())})
        matrix = ConfusionMatrix(names=names)
        for preds, targets in records:
            matrix.process_batch(preds, targets, conf=selected, iou_thres=0.50)
        matrix.plot(normalize=False, save_dir=out)
        matrix.plot(normalize=True, save_dir=out)
        with (out / 'confusion_matrix_counts.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['predicted / actual', *names.values(), 'background'])
            writer.writerows([name, *map(int, row)] for name, row in zip([*names.values(), 'background'], matrix.matrix))
        mean = np.asarray(values).mean(0)
        summary[split] = {'mAP50': float(ap[:, 0].mean()), 'mAP50_95': float(ap.mean()),
                          'macro_precision_iou50': float(mean[0]), 'macro_recall_iou50': float(mean[1]),
                          'macro_f1_iou50': float(mean[2]), 'images': len(records),
                          'gt_instances': len(stats[3]), 'classes_evaluated': 18,
                          'annotation_sha256': metadata['annotation_sha256'],
                          'prediction_cache_sha256': metadata['prediction_cache_sha256'],
                          'confusion_matrix': {'confidence': selected, 'matching_iou': 0.50}}
    summary['weights_sha256'] = weights_sha
    baseline = json.loads(args.baseline.read_text())
    ap_delta = 100 * (summary['test']['mAP50'] - baseline['test']['mAP50'])
    f1_delta = 100 * (summary['test']['macro_f1_iou50'] - baseline['test']['macro_f1_iou50'])
    summary['comparison'] = {'baseline': 'YOLO26s', 'baseline_weights_sha256': baseline['weights_sha256'],
                             'baseline_test': baseline['test'], 'mAP50_delta_percentage_points': ap_delta,
                             'macro_f1_delta_percentage_points': f1_delta,
                             'both_submission_metrics_better': bool(ap_delta > 0 and f1_delta > 0),
                             'different_inference_preprocessing': 'RF-DETR square resize; YOLO default letterbox.',
                             'uncertainty': 'Single-seed experiment; no statistical superiority claim.'}
    with (args.output / 'per_class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
