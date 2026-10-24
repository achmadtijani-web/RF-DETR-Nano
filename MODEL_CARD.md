# RF-DETR Nano — model card

RF-DETR Nano unggul pada kedua metrik submission.

Bobot ini berasal dari arsitektur RF-DETR Nano, bukan model YOLO yang diganti nama.
RF-DETR 1.9.0; Torch 2.7.1+cu128; 12 epoch aktual;
640 × 640; batch 8; gradient accumulation 2; seed 42; gradient checkpointing.
Checkpoint terbaik: epoch 12, dipilih berdasarkan validation EMA COCO mAP@50–95.

| Test internal: 1.059 gambar, 2.669 objek, 18 kelas | Hasil |
|---|---:|
| mAP@50 | 72.12% |
| macro-F1 pada IoU 0,50 | 71.35% |
| mAP@50–95 | 55.43% |

Confidence 0.3943943943943944 dipilih dari maksimum rata-rata F1 per kelas
di validation dan dipakai tanpa perubahan di test. Evaluasi AP menggunakan confidence
minimum 0,001, maksimal 300 prediksi per gambar, pencocokan greedy IoU dan kurva AP/F1
Ultralytics 8.4.182 yang sama dengan YOLO26s. RF-DETR menggunakan square resize,
tanpa NMS, dan bounding box dipotong ke batas gambar. Baseline memakai preprocessing
dan NMS bawaannya. Metrik COCO saat training dapat berbeda dari metrik submission ini.

Selisih terhadap YOLO26s: mAP@50 +5.33
poin persentase; macro-F1 +4.04
poin persentase. Ini satu seed dan bukan klaim keunggulan statistik.

Train 8.822/18.002 objek; validation 2.046/4.758; test 1.059/2.669. Kelas mirip
tetap terpisah sesuai anotasi penyelenggara. Audit awal dan hash anotasi dipertahankan.
Test internal telah digunakan dalam eksperimen sebelumnya; model belum divalidasi
untuk diagnosis agronomi atau kamera drone di lapangan independen. Periksa AP per
kelas dan confusion matrix sebelum penggunaan. Dataset asli tidak didistribusikan.

SHA256 RF-DETR-Nano.pth: `16986ca5759c80869dd46fe121efd1ae434a98a1831dc8bf86d27b6dacb38338`.
Semua tensor cocok dengan checkpoint hasil training sebelum metadata dibersihkan.

Lihat RFDETR_LICENSE.txt untuk lisensi RF-DETR dan THIRD_PARTY_NOTICES.md untuk dependensi.
