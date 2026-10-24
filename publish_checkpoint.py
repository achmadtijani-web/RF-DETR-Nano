"""Make a portable inference checkpoint while preserving every trained tensor."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def clean(value, key=''):
    if key in {'dataset_dir', 'coco_path'}:
        return 'data/coco'
    if key == 'output_dir':
        return 'runs/nano_640_seed42'
    if key in {'resume', 'pretrain_weights', 'pretrained_encoder'}:
        return None
    if isinstance(value, argparse.Namespace):
        return argparse.Namespace(**{k: clean(v, k) for k, v in vars(value).items()})
    if isinstance(value, dict):
        return {str(k): clean(v, str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return clean(str(value), key)
    if isinstance(value, str) and re.match(r'^(?:[A-Za-z]:[\\/]|/Users/|/home/)', value):
        return Path(value.replace('\\', '/')).name
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/RF-DETR-Nano.pth')
    args = parser.parse_args()
    import torch
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    assert isinstance(original.get('model'), dict)
    config = clean(original.get('model_config') or json.loads(
        (ROOT / 'reports/requested_config.json').read_text())['model'])
    config['model_name'] = 'RFDETRNano'
    original_args = original['args']
    original_args = vars(original_args) if isinstance(original_args, argparse.Namespace) else original_args
    # v1.9's training callback may serialize TrainConfig alone; preserve architectural
    # query-group metadata explicitly so inference never guesses how to slice query tensors.
    portable_args = {**config, **clean(original_args)}
    payload = {'model': {key: value.detach().cpu() for key, value in original['model'].items()},
               'args': portable_args, 'model_config': config,
               'model_name': 'RFDETRNano', 'rfdetr_version': original.get('rfdetr_version', '1.9.0'),
               'epoch': original.get('epoch'), 'global_step': original.get('global_step'),
               'submission_model_name': 'RF-DETR Nano'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite an existing published checkpoint.')
    torch.save(payload, args.output)
    saved = torch.load(args.output, map_location='cpu', weights_only=False)
    assert set(original['model']) == set(saved['model'])
    assert all(torch.equal(tensor, saved['model'][key]) for key, tensor in original['model'].items())
    report = {'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'weights': 'models/RF-DETR-Nano.pth', 'weights_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'all_model_tensors_identical': True, 'tensor_count': len(saved['model']),
              'trained_epoch_zero_based': original.get('epoch'),
              'note': 'Only metadata and redundant training state removed; every model tensor is unchanged.'}
    (ROOT / 'reports/public_checkpoint.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
