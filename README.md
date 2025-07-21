# 🗑️ SiPilah (Sosialisasi Cerdas Pengelolaan Sampah)

Proyek ini membangun model klasifikasi gambar sampah menggunakan **Convolutional Neural Network (CNN)** berbasis **MobileNetV2** dan menyediakan antarmuka pengguna interaktif melalui **Streamlit**.

---

## 📌 Deskripsi

Model ini dapat mengklasifikasikan gambar sampah ke dalam beberapa kategori seperti:

- B3 (Bahan Berbahaya dan Beracun)
- Daur Ulang
- Organik
- Residu

## 📁 Struktur Proyek

├── dataset/ # Dataset yang dibagi menjadi train/val/test
│ ├── train/
│ ├── test/
│ └── val/
│
├── models/ # Folder model terlatih
│ ├── model.h5 # Model Keras
│ └── tfjs_model/ # Model TensorFlow.js
│ ├── model.json
│ ├── group1-shard1of3.bin
│ ├── group1-shard2of3.bin
│ └── group1-shard3of3.bin
│
├── notebooks/
│ └── training.ipynb # Notebook pelatihan model
│
├── app.py # Aplikasi Streamlit untuk prediksi
├── train.py # Script pelatihan model via terminal
├── requirements.txt # Dependensi Python
└── README.md # Dokumentasi proyek ini

## Instalasi dan Persiapan

```
git clone https://github.com/MHusni1604/KKN-Periode-2.git
cd KKN-Periode-2
pip install -r requirements.txt
```

## Cara Melatih Model

```
python training.py
```

## Cara Menggunakan Model / Aplikasi

```
streamlit run app.py
```

## Arsitektur Model

Model dikembangkan menggunakan arsitektur MobileNetV2 dengan pre-trained weights dari ImageNet, dan dilakukan fine-tuning untuk klasifikasi sampah ke dalam 4 kelas (b3, daur ulang, organik, residu).

```
Input (150x150x3)
↓
MobileNetV2 (tanpa fully connected layer)
↓
GlobalAveragePooling2D
↓
Dense (128 neuron, ReLU)
↓
BatchNormalization
↓
Dropout (rate=0.5)
↓
Dense (softmax, jumlah neuron = jumlah kelas)
```

Augmentasi diterapkan hanya pada data pelatihan untuk meningkatkan generalisasi model:
- Rotasi acak: rotation_range=10
- Translasi horizontal & vertikal: width_shift_range=0.1, height_shift_range=0.1
- Shear dan Zoom: shear_range=0.1, zoom_range=0.1
- Normalisasi: rescale=1./255

Proses pelatihan dilakukan dengan:
- Optimizer: Adam, learning_rate=1e-5
- Loss: categorical_crossentropy
- Metric: accuracy
- Epochs: 100 (dengan early stopping)
- Batch size: 32
- Callback:
  - EarlyStopping (patience 5)
  - ModelCheckpoint (menyimpan model terbaik)
  - Custom Callback (StopTrainingAtAccuracy) untuk menghentikan pelatihan otomatis jika akurasi dan val_akurasi > 95%
 
## Evaluasi Model
Dari model yang terbentuk diperolehh akurasi pada dataset test sebesar 94,91%
