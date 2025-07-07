import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# --- Load model ---
@st.cache_resource
def load_cnn_model():
    model = load_model('C:/Users/acer/Desktop/KKN-Periode-2/models/model.h5')
    return model

model = load_cnn_model()

# --- Daftar kelas ---
class_names = ['b3', 'daur_ulang', 'organik', 'residu']

# --- Judul ---
st.title("🗑️ Klasifikasi Sampah dengan Deep Learning")
st.write("Upload gambar atau gunakan kamera, lalu model akan mengklasifikasikan ke dalam salah satu kategori.")

# --- Pilih metode input gambar ---
input_method = st.radio("Pilih metode input gambar:", ["Upload File", "Gunakan Kamera"])

# --- Ambil gambar sesuai metode ---
image = None
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
elif input_method == "Gunakan Kamera":
    camera_image = st.camera_input("Ambil gambar sampah")
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')

# --- Jika ada gambar, tampilkan dan klasifikasikan ---
if image is not None:
    st.image(image, caption='Gambar yang dipilih', use_column_width=True)

    # Preprocessing
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.markdown("---")
    st.subheader("📋 Hasil Prediksi")
    st.markdown(f"**Kategori:** {class_names[class_index]}")
    st.markdown(f"**Tingkat keyakinan:** {confidence:.2%}")
