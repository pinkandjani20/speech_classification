# Klasifikasi Suara Hewan Imitasi dengan CNN

Proyek ini mendemonstrasikan cara mengklasifikasikan 5 jenis suara hewan imitasi ("moo", "meow", "woof", "mbee", "tweet") menggunakan Convolutional Neural Network (CNN) di PyTorch. Dataset dibuat dengan merekam atau mensimulasikan suara hewan.

## Dataset
- 5 kelas: moo (sapi), meow (kucing), woof (anjing), mbee (kambing), tweet (burung)
- Setiap kelas berisi 15 sampel audio (10 untuk training, 5 untuk testing)
- Audio direkam atau disimulasikan dan disimpan sebagai file .wav di folder `data/`

## Alur Kerja
1. **Persiapan Data**: Rekam atau simulasikan file .wav untuk setiap kelas hewan
2. **Preprocessing**: Load audio, normalisasi, ekstraksi fitur MFCC
3. **Dataset & DataLoader**: Dataset PyTorch kustom untuk memuat fitur dan label
4. **Model**: CNN sederhana untuk klasifikasi
5. **Training & Evaluasi**: Melatih model dan mengevaluasi akurasi

## Insight Notebook

### 1. Import Library & Parameter
Cell ini melakukan import seluruh library yang dibutuhkan dan mendefinisikan parameter utama seperti sample rate, durasi audio, jumlah sample, serta daftar kelas suara hewan. Semua proses selanjutnya akan bergantung pada parameter dan library yang diinisialisasi di sini.

### 2. Perekaman Data
Cell ini berfungsi untuk merekam data audio dari setiap kelas hewan (moo, meow, woof, mbee, tweet) sebanyak 15 kali per kelas. Hasil rekaman akan digunakan sebagai dataset utama untuk proses training dan evaluasi model.

### 3. Preprocessing, Ekstraksi Fitur, Dataset, DataLoader
Cell ini melakukan preprocessing audio (VAD, normalisasi, padding/cropping), ekstraksi fitur MFCC, serta membangun custom dataset dan DataLoader. Hasilnya adalah data audio yang sudah siap digunakan untuk training dan evaluasi model CNN.

### 4. Visualisasi Gelombang & MFCC
Visualisasi ini memperlihatkan bentuk gelombang asli dan hasil preprocessing, serta MFCC dari masing-masing kelas suara hewan. Dari grafik ini, kita dapat melihat perbedaan karakteristik audio tiap kelas yang akan membantu model dalam proses klasifikasi.

### 5. Analisis Formant (Opsional)
Analisis formant pada cell ini mencoba membandingkan spektrum frekuensi suara hewan dengan rentang teoritis formant. Namun, karena suara hewan berbeda dengan vokal manusia, hasil deteksi formant lebih bersifat ilustratif dan tidak selalu relevan untuk klasifikasi hewan.

### 6. Training Model
Cell ini melakukan training model CNN untuk klasifikasi suara hewan. Grafik loss dan akurasi menunjukkan seberapa baik model belajar dari data. Jika loss menurun dan akurasi meningkat, berarti model berhasil mempelajari pola dari data training.

### 7. Evaluasi Model
Cell ini mengevaluasi performa model pada data test. Akurasi dan classification report menunjukkan seberapa baik model membedakan suara hewan pada data yang belum pernah dilihat. Nilai precision, recall, dan f1-score yang tinggi menandakan model sangat baik dalam klasifikasi.

### 8. Penyimpanan Model
Model yang sudah terlatih disimpan ke file agar bisa digunakan kembali untuk prediksi suara hewan tanpa perlu training ulang. Ini memudahkan deployment dan pengujian model di masa depan.

### 9. Uji Coba Prediksi
Cell ini menguji model dengan merekam suara hewan baru dan menampilkan prediksi beserta confidence untuk setiap kelas. Hasil prediksi membantu mengetahui seberapa yakin model dalam mengklasifikasikan suara yang diberikan.

## Cara Menjalankan
1. Install dependensi:
   ```
   pip install torch torchvision torchaudio 
   pip install librosa sounddevice soundfile matplotlib scikit-learn
   ```
2. Jalankan notebook `vocal.ipynb` langkah demi langkah

## Catatan
- Untuk penggunaan nyata, rekam suara hewan menggunakan mikrofon.
- Kode ini modular dan dapat diadaptasi untuk tugas klasifikasi audio lainnya.