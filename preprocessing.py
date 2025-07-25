import cv2
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Membaca gambar dari path, resize, normalisasi,
    dan mengembalikan array numpy siap input CNN.
    """
    # Baca gambar dari file (format BGR)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal membaca gambar dari path: {image_path}")
    
    # Resize gambar ke ukuran target (default 128x128)
    image_resized = cv2.resize(image, target_size)
    
    # Konversi BGR ke RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalisasi pixel ke [0,1]
    image_norm = image_rgb.astype('float32') / 255.0
    
    # Tambahkan dimensi batch (1, height, width, channels)
    image_input = np.expand_dims(image_norm, axis=0)
    
    return image_input
