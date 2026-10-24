"""Export the exact Ultralytics-cleaned labels to RF-DETR's COCO layout."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-yolo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    from ultralytics.data.dataset import YOLODataset
    mapping = json.loads((args.source_yolo / 'class_mapping.json').read_text())['classes']
    categories = [dict(id=x['yolo_id'], name=x['name'], supercategory='none') for x in mapping]
    assert [x['id'] for x in categories] == list(range(18))
    expected = json.loads((ROOT / 'reports/dataset_manifest.json').read_text())
    assert expected['categories'] == categories
    output = args.output.resolve()
    if not args.verify_only and output.exists() and any(output.iterdir()):
        raise RuntimeError('Choose a new empty output to avoid stale images or labels.')
    result = {'scope': 'full', 'target': str(output), 'categories': categories,
              'counts': {}, 'test_used_for_training': False}
    for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        cache = args.source_yolo / 'labels' / f'{split}.cache'
        if cache.exists():
            records = np.load(cache, allow_pickle=True).item()['labels']
        else:
            dataset = YOLODataset(img_path=str(args.source_yolo / 'images' / split),
                                  data={'names': {x['id']: x['name'] for x in categories}, 'nc': 18},
                                  task='detect', augment=False, imgsz=640, batch_size=8, cache=False)
            records = dataset.labels
        images = []; annotations = []
        for index, record in enumerate(records):
            image = Path(record['im_file'])
            height, width = record['shape']
            images.append(dict(id=index, file_name=image.name, width=width, height=height))
            for cid, box in zip(record['cls'].reshape(-1), record['bboxes']):
                cx, cy, bw, bh = map(float, box)
                bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
                annotations.append(dict(id=len(annotations) + 1, image_id=index, category_id=int(cid),
                                        bbox=bbox, area=bbox[2] * bbox[3], iscrowd=0))
        payload = dict(info={'description': 'Same cleaned Telepati18 classes; loader-deduplicated annotations'},
                       licenses=[], images=images, annotations=annotations, categories=categories)
        encoded = json.dumps(payload, separators=(',', ':')).encode()
        counts = dict(images=len(images), annotations=len(annotations),
                      annotation_sha256=hashlib.sha256(encoded).hexdigest())
        assert counts == expected['counts'][split], (split, counts, expected['counts'][split])
        result['counts'][split] = counts
        if not args.verify_only:
            destination = output / folder
            destination.mkdir(parents=True)
            (destination / '_annotations.coco.json').write_bytes(encoded)
            for record in records:
                source = Path(record['im_file'])
                try:
                    os.link(source, destination / source.name)
                except OSError:
                    shutil.copy2(source, destination / source.name)
        print(split, json.dumps(counts), flush=True)
    if not args.verify_only:
        (output / 'dataset_reference.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('All three annotation hashes match the original cleaned experiment.', flush=True)


if __name__ == '__main__':
    main()
