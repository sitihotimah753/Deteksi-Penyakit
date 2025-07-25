from flask import Flask, render_template, request, send_from_directory
import os
from utils.helpers import handle_prediction, DESCRIPTIONS

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route untuk menampilkan file gambar yang diupload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menangani prediksi
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    if not file or file.filename == '':
        return render_template('result.html', 
                               label="Tidak ada file yang dipilih.",
                               confidence=0,
                               damage_level_str="N/A",
                               severity_numeric=0,
                               filename=None,
                               deskripsi={
                                   "deskripsi": "Tidak ada deskripsi.",
                                   "solusi": "Tidak ada solusi."
                               })

    # Proses prediksi dan ekstraksi informasi hasil
    result = handle_prediction(file, app.config['UPLOAD_FOLDER'])
    return render_template('result.html', **result)

# Menjalankan aplikasi
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)

