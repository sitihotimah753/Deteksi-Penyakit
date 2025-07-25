import os
from cnn_model import predict_class
from fuzzy_system import get_damage_level

# Deskripsi penyakit tebu dan solusi penanganannya
DESCRIPTIONS = {
    'sehat': {
        'deskripsi': 'Daun tebu sehat, warna hijau cerah, tidak ada bercak.',
        'solusi': 'Pertahankan pemupukan, irigasi, dan pengendalian gulma.'
    },
    'mosaik': {
        'deskripsi': 'Bercak mosaik kuning/hijau muda pada daun.',
        'solusi': 'Gunakan bibit sehat dan kendalikan vektor virus.'
    },
    'busuk merah': {
        'deskripsi': 'Lesi merah pada daun dan pelepah, batang memerah.',
        'solusi': 'Gunakan varietas resisten dan lakukan rotasi tanaman.'
    },
    'karat': {
        'deskripsi': 'Pustula oranye/coklat di permukaan daun.',
        'solusi': 'Gunakan varietas resisten, tingkatkan sirkulasi udara.'
    },
    'kuning': {
        'deskripsi': 'Daun menguning, bisa karena nutrisi atau stres lingkungan.',
        'solusi': 'Analisa tanah dan kelola drainase serta irigasi.'
    }
}

def handle_prediction(file, upload_folder):
    """
    Menangani proses prediksi dari file gambar yang diunggah.

    Args:
        file: Objek file dari form.
        upload_folder: Direktori penyimpanan sementara.

    Returns:
        dict: Informasi prediksi lengkap.
    """
    result = {
        'label': "Tidak Dikenali",
        'confidence': 0.0,
        'damage_level_str': "N/A",
        'severity_numeric': 0,
        'filename': None,
        'deskripsi': {'deskripsi': '-', 'solusi': '-'}  # fallback default
    }

    if not file or file.filename == '':
        result['label'] = "Tidak ada file yang dipilih."
        return result

    try:
        # Simpan file ke folder upload
        filename = file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        result['filename'] = filename

        # Prediksi dengan CNN
        predicted_label_raw, confidence = predict_class(filepath)

        if "Error" in predicted_label_raw:
            result['label'] = predicted_label_raw
            return result

        # Bersihkan dan normalisasi label
        predicted_label = predicted_label_raw.lower().strip()

        # Simpan label dan confidence
        result['label'] = predicted_label
        result['confidence'] = round(confidence * 100, 2)  # dari 0.85 → 85.0%

        # Prediksi tingkat kerusakan (dari sistem fuzzy)
        damage_level_str = get_damage_level(predicted_label)
        result['damage_level_str'] = damage_level_str
        result['severity_numeric'] = parse_severity(damage_level_str)

        # Deskripsi penyakit dan solusi
        result['deskripsi'] = DESCRIPTIONS.get(predicted_label, DESCRIPTIONS['sehat'])

    except Exception as e:
        result['label'] = f"Terjadi kesalahan saat memproses gambar: {e}"

    return result

def parse_severity(damage_str):
    """
    Mengubah string tingkat kerusakan (misal: '75%') menjadi angka (float).

    Args:
        damage_str (str): Nilai kerusakan dalam string.

    Returns:
        float: Nilai kerusakan dalam angka (tanpa %).
    """
    try:
        if '%' in damage_str:
            return float(damage_str.replace('%', '').strip())
        elif '-' in damage_str:
            parts = damage_str.replace('%', '').split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(damage_str.strip())
    except Exception:
        return 0.0
