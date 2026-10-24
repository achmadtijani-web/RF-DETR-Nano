"""Train a genuine RF-DETR Nano on the unchanged cleaned Telepati split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.15 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-reference', type=Path, required=True)
    parser.add_argument('--pretrain', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--run-name', default='nano_640_seed42')
    args = parser.parse_args()
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    if Path(args.run_name).name != args.run_name:
        raise ValueError('--run-name must be a single directory name.')
    output = ROOT / 'runs' / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    report = ROOT / 'reports/training_status.json'
    previous = json.loads(report.read_text()) if report.exists() else {}
    if previous.get('status') == 'complete' and not args.resume:
        raise RuntimeError('Training already completed; use a new run or explicit --resume.')
    started = time.perf_counter()
    state = {'status': 'starting', 'model': 'RF-DETR Nano',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'requested_epochs': args.epochs,
             'test_used_for_training': False, 'output_dir': str(output)}

    def update(**values):
        state.update(values)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds'] = time.perf_counter() - started
        try:
            atomic_json(report, state)
        except OSError as exc:
            print(f'Nonfatal progress-write error: {exc}', flush=True)

    update(phase='imports')
    try:
        os.environ.setdefault('RF_HOME', str(ROOT / '.cache'))
        os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
        import importlib.metadata
        import torch
        from pytorch_lightning import Callback, seed_everything
        from rfdetr.config import RFDETRNanoConfig, TrainConfig
        from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer

        torch.set_num_threads(4)
        seed_everything(42, workers=True)
        reference = json.loads(args.data_reference.read_text(encoding='utf-8'))
        data = Path(reference['target'])
        for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
            annotation = data / folder / '_annotations.coco.json'
            encoded = annotation.read_bytes()
            assert hashlib.sha256(encoded).hexdigest() == reference['counts'][split]['annotation_sha256']
            payload = json.loads(encoded)
            assert len(payload['images']) == reference['counts'][split]['images']
            assert len(payload['annotations']) == reference['counts'][split]['annotations']
            assert payload['categories'] == reference['categories']
        names = [x['name'] for x in sorted(reference['categories'], key=lambda x: x['id'])]
        assert [x['id'] for x in reference['categories']] == list(range(18))
        assert [reference['counts'][s]['images'] for s in ('train', 'val', 'test')] == [8822, 2046, 1059]
        assert args.pretrain.is_file()
        atomic_json(ROOT / 'reports/dataset_reference.json', reference)
        update(phase='initializing', dataset_images={'train': 8822, 'val': 2046, 'test': 1059},
               gpu=torch.cuda.get_device_name(0), rfdetr=importlib.metadata.version('rfdetr'),
               torch=torch.__version__, batch_size=8, grad_accum_steps=2, resolution=640,
               checkpoint_selection='Maximum validation EMA COCO mAP@50-95; no test selection.')

        class Progress(Callback):
            def on_train_epoch_start(self, trainer, module):
                self.epoch_start = time.perf_counter()
                update(status='running', phase='train', epoch=trainer.current_epoch + 1,
                       batch=0, batches=int(trainer.num_training_batches))

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                if batch_idx % 25 == 0:
                    update(phase='train', epoch=trainer.current_epoch + 1,
                           batch=batch_idx + 1, batches=int(trainer.num_training_batches),
                           epoch_elapsed_seconds=time.perf_counter() - self.epoch_start,
                           peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20)

            def on_validation_start(self, trainer, module):
                # Retain optimizer/EMA state before external metric/rendering code runs.
                # This is a recovery file only, never a candidate selected without validation.
                if not trainer.sanity_checking:
                    trainer.save_checkpoint(output / 'pre_validation.ckpt')
                update(phase='validation', epoch=trainer.current_epoch + 1, validation_batch=0)

            def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
                if batch_idx % 25 == 0:
                    update(phase='validation', validation_batch=batch_idx + 1)

            def on_validation_end(self, trainer, module):
                metrics = {}
                for key, value in trainer.callback_metrics.items():
                    if str(key).startswith('val/') and hasattr(value, 'numel') and value.numel() == 1:
                        metrics[str(key)] = float(value.detach().cpu())
                result = {'epoch': trainer.current_epoch + 1, 'metrics': metrics,
                          'elapsed_seconds': time.perf_counter() - started}
                with (ROOT / 'reports/epoch_history.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(result) + '\n')
                print('EPOCH_RESULT ' + json.dumps(result), flush=True)
                update(phase='epoch_complete', completed_epochs=trainer.current_epoch + 1,
                       latest_validation=metrics)

        model_config = RFDETRNanoConfig(
            num_classes=18, resolution=640, positional_encoding_size=40,
            gradient_checkpointing=True, device='cuda', pretrain_weights=str(args.pretrain.resolve()))
        train_config = TrainConfig(
            dataset_dir=str(data), dataset_file='roboflow', output_dir=str(output),
            epochs=args.epochs, batch_size=8, grad_accum_steps=2,
            lr=1e-4, lr_encoder=1.5e-4, lr_drop=max(1, args.epochs - 2),
            num_workers=args.workers, multi_scale=False, expanded_scales=False, scale_jitter=False,
            # RF-DETR 1.9.0's COCO callback returns early with eval_ema_only=True
            # because the regular accumulator is empty. Evaluate both tracks so
            # validation metrics and EMA checkpoint selection are actually computed.
            use_ema=True, eval_ema_only=False, eval_max_dets=300, compute_val_loss=False,
            checkpoint_interval=4, tensorboard=False, wandb=False, run_test=False,
            progress_bar=None, class_names=names, seed=42,
            resume=str(args.resume.resolve()) if args.resume else None,
            notes={'purpose': 'Full-data RF-DETR Nano comparison', 'test_selection': False})
        atomic_json(ROOT / 'reports/requested_config.json', {
            'model': model_config.model_dump(mode='json'), 'train': train_config.model_dump(mode='json')})
        module = RFDETRModelModule(model_config, train_config)
        datamodule = RFDETRDataModule(model_config, train_config)
        trainer = build_trainer(train_config, model_config, accelerator='gpu', devices=1,
                                num_sanity_val_steps=0)
        trainer.callbacks.append(Progress())
        torch.cuda.reset_peak_memory_stats()
        trainer.fit(module, datamodule, ckpt_path=train_config.resume)
        assert 'val/ema_mAP_50_95' in state.get('latest_validation', {}), (
            'No EMA validation metrics were recorded; cannot claim validation-based checkpoint selection.')
        # Explicit EMA selection: total may select the regular model instead.
        best = output / 'checkpoint_best_ema.pth'
        assert best.is_file(), 'Best checkpoint was not produced.'
        update(status='complete', phase='trained', completed_epochs=trainer.current_epoch,
               best_checkpoint=str(best), best_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
               peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as exc:
        update(status='failed', phase='exception', error=f'{type(exc).__name__}: {exc}',
               traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
