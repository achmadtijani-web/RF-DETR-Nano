"""Check correct/wrong-class/duplicate/background detections against known answers."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    from score import load_split
    import numpy as np
    folder = ROOT / 'reports/metric_protocol_check'
    annotation_folder = folder / 'dataset/valid'
    annotation_folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'cache'
    cache.mkdir(exist_ok=True)
    categories = [dict(id=i, name=f'class_{i}', supercategory='none') for i in range(18)]
    annotation = {'categories': categories,
                  'images': [{'id': 0}, {'id': 1}],
                  'annotations': [{'image_id': 0, 'category_id': 0, 'bbox': [0, 0, 10, 10], 'iscrowd': 0}]}
    encoded = json.dumps(annotation).encode()
    (annotation_folder / '_annotations.coco.json').write_bytes(encoded)
    rows = [{'image_id': 0, 'xyxy': [[0, 0, 10, 10]] * 3,
             'confidence': [0.9, 0.8, 0.7], 'class_id': [1, 0, 0]},
            {'image_id': 1, 'xyxy': [[0, 0, 10, 10]], 'confidence': [0.6], 'class_id': [0]}]
    with gzip.open(cache / 'val.jsonl.gz', 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    metadata = {'complete': True, 'partial_smoke_test': False, 'images': 2,
                'weights_sha256': 'synthetic_known_answer', 'max_det': 300, 'confidence_floor': 0.001,
                'annotation_sha256': hashlib.sha256(encoded).hexdigest(),
                'prediction_cache_sha256': hashlib.sha256((cache / 'val.jsonl.gz').read_bytes()).hexdigest()}
    (cache / 'val_metadata.json').write_text(json.dumps(metadata))
    stats, _, _ = load_split(folder / 'dataset', cache, 'val', {c['id']: c['name'] for c in categories})
    assert stats[0].shape == (4, 10)
    assert stats[0][:, 0].tolist() == [False, True, False, False]
    assert np.all(stats[0][1]) and not np.any(stats[0][[0, 2, 3]])
    assert stats[3].tolist() == [0]
    report = {'status': 'passed', 'predictions': 4, 'gt_objects': 1,
              'correct_detections_at_all_ten_IoUs': 1,
              'cases': ['wrong class is FP', 'correct box is TP', 'duplicate prediction is FP',
                        'prediction on background image is FP']}
    (ROOT / 'reports/metric_protocol_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
