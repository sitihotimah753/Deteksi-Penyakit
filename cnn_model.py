# cnn_model.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Path ke model Anda. PENTING: Untuk aplikasi web, gunakan path relatif
# atau pastikan model berada di direktori yang dapat diakses oleh aplikasi Flask.
# Saya akan menggunakan nama file saja, asumsikan model ini ada di direktori yang sama
# dengan app.py atau di subfolder 'models'.
MODEL_PATH = 'D:/tugasakhir/model_cnn_tebu.h5' # <--- Pastikan nama file model Anda benar

# Nama kelas harus sesuai urutan output model Anda
class_names = ['sehat', 'mosaik', 'busuk merah', 'karat', 'kuning']

# Muat model di awal agar tidak perlu memuat setiap kali prediksi
try:
    model = load_model(MODEL_PATH)
    # Tidak perlu model.build() jika model sudah dimuat dari file .h5
    print(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"Error: Gagal memuat model dari '{MODEL_PATH}'. Pastikan file ada dan benar. Error: {e}")
    model = None # Set model ke None jika gagal dimuat

def predict_class(filepath):
    """
    Melakukan prediksi kelas penyakit dari gambar yang diberikan.

    Args:
        filepath (str): Path lengkap ke file gambar.

    Returns:
        tuple: (label_penyakit, tingkat_kepercayaan)
               label_penyakit (str): Nama kelas yang diprediksi.
               tingkat_kepercayaan (float): Probabilitas kelas yang diprediksi.
    """
    if model is None:
        return "Error: Model tidak dimuat.", 0.0

    if not os.path.exists(filepath):
        return f"Error: File gambar tidak ditemukan di path: {filepath}", 0.0

    try:
        # Muat gambar dan ubah ukurannya sesuai target_size model Anda
        # Pastikan target_size sesuai dengan input model Anda (misal: 128x128)
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
        img_array /= 255.0 # Normalisasi piksel jika model dilatih dengan input 0-1

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        return label, confidence

    except Exception as e:
        return f"Error saat memproses gambar atau prediksi: {e}", 0.0

# Fungsi extract_features dan feature_model dihapus
# karena tidak digunakan untuk penentuan tingkat kerusakan yang diinginkan.
