# RF-DETR Nano — Smart Agriculture

RF-DETR Nano unggul pada kedua metrik submission.

Model RF-DETR Nano asli untuk 18 label tanaman Telepati 8.0, dilatih penuh
12 epoch pada input 640 × 640, batch 8, akumulasi
gradien 2, seed 42. Bobot awal RF-DETR Nano resmi; arsitektur ini berbeda dari YOLO.

| Model | mAP@50 test | macro-F1 test |
|---|---:|---:|
| YOLO26s | 66.80% | 67.31% |
| RF-DETR Nano | 72.12% | 71.35% |

Angka berasal dari test internal yang sama (1.059 gambar/2.669 objek).
Confidence 0.3943943943943944 dipilih dari validation, bukan test.
Checkpoint dipilih melalui validation EMA COCO mAP@50–95.
Lihat MODEL_CARD.md, laporan per kelas, dan confusion matrix untuk definisi serta batas hasil.
Test internal ini juga digunakan dalam eksperimen sebelumnya; satu seed tidak
membuktikan keunggulan statistik atau kinerja agronomi/drone di lapangan.

## Berkas

- RF_DETR_Nano_Experiments.ipynb: notebook dengan keluaran analisis CPU nyata.
- models/RF-DETR-Nano.pth: letakkan bobot fine-tuning yang diunduh dari Release di sini.
- reports/evaluation/metrics.json dan per_class_metrics.csv: skor dan 18 kelas.
- reports/training_summary.json: epoch aktual dan riwayat validation.
- reports/dataset_manifest.json: jumlah gambar/objek dan hash anotasi tiga split.
- reports/public_checkpoint.json: hash bobot dan bukti semua tensor tidak berubah.

## Reproduksi

Gunakan Python 3.11 dan Torch 2.7.1/Torchvision 0.22.1 dari indeks resmi
yang sesuai perangkat, lalu instal requirements.txt. requirements-rfdetr-lock.txt
merekam lingkungan RF-DETR eksperimen; metrik memakai Ultralytics 8.4.182.

Dataset tidak disertakan. Sediakan ekspor COCO asli penyelenggara di
data/raw/telepati. Jangan mengubah pembagian train/validation/test.

```powershell
python prepare_dataset_yolo.py --source data/raw/telepati --output data/yolo --reports reports/reproduction_audit
python prepare_coco.py --source-yolo data/yolo --output data/coco
python train.py --data-reference data/coco/dataset_reference.json --pretrain models/rf-detr-nano.pth --epochs 12
```

models/rf-detr-nano.pth pada perintah training adalah checkpoint **awal resmi**
yang diunduh melalui RF-DETR, bukan bobot akhir RF-DETR-Nano.pth dari Release.
Untuk memulai dari checkpoint akhir, perlakukan sebagai eksperimen warm start baru.

```powershell
python cache_predictions.py --weights models/RF-DETR-Nano.pth --dataset data/coco --output reports/predictions_reproduced
python score.py --dataset data/coco --cache reports/predictions_reproduced --baseline reports/baseline_yolo26s.json --output reports/evaluation_reproduced
```

Untuk inference gambar sendiri, buka notebook dan aktifkan RUN_INFERENCE setelah
menempatkan gambar di examples/input.jpg. Run All bawaan hanya menjalankan analisis
laporan; instalasi, persiapan data, training, dan inference dinonaktifkan sampai flag diaktifkan.

RF-DETR memakai square resize dan tanpa NMS. Slot no-object tidak diperlakukan
sebagai kelas dataset. Maksimal 300 deteksi/gambar, confidence minimum AP 0,001.
Implementasi AP/F1 dan pencocokan IoU sama dengan baseline; preprocessing kedua
arsitektur mengikuti kebiasaannya sehingga seluruh faktor tidak terisolasi.

SHA256 bobot: `16986ca5759c80869dd46fe121efd1ae434a98a1831dc8bf86d27b6dacb38338`.
Sumber: [RF-DETR](https://github.com/roboflow/rf-detr),
[panduan resmi](https://rfdetr.roboflow.com/learn/train/).
Lihat THIRD_PARTY_NOTICES.md dan berkas lisensi dependensi.
