from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Memuat model YOLO satu kali
model = YOLO('models/bestt.pt')

def clear_old_files(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def detect_karang(input_image_path, output_image_path):
    results = model(input_image_path)
    detected_image = results[0].plot()
    detected_image_pil = Image.fromarray(detected_image)
    detected_image_pil.save(output_image_path)

def analyze_karang(input_image_path, output_image_path):
    results = model(input_image_path)
    masks = results[0].masks
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

    if masks is None:
        raise ValueError("Model tidak mendukung segmentasi atau tidak ada mask yang ditemukan.")

    total_pixels = 0
    karang_pixels = 0
    class_percentage = {}

    class_names = {0: "Bukan Karang", 1: "Karang"}

    for mask in masks.data:
        mask_np = mask.cpu().numpy()
        total_pixels += mask_np.size
        karang_pixels += np.sum(mask_np == 1)
        unique_classes = np.unique(mask_np)

        for cls in unique_classes:
            if cls not in class_percentage:
                class_percentage[cls] = 0
            class_percentage[cls] += np.sum(mask_np == cls)

    percentage_karang = (karang_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    detected_image = results[0].plot()
    cv2.imwrite(output_image_path, detected_image)

    class_percentage = {class_names.get(int(key), f"Class {key}"): int(value) for key, value in class_percentage.items()}
    labels = list(class_percentage.keys())
    sizes = list(class_percentage.values())
    colors = plt.cm.Paired(np.linspace(0, 1, len(class_percentage)))

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.axis('equal')

    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    diagram_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close()

    analysis_data = {
        'total_pixels': int(total_pixels),
        'karang_pixels': int(karang_pixels),
        'percentage_karang': float(percentage_karang),
        'class_percentage': class_percentage,
        'detected_classes': detected_classes  # Tambahkan kelas yang terdeteksi
    }

    return diagram_base64, analysis_data

@app.route('/analyze', methods=['POST'])
def analyze():
    clear_old_files(app.config['UPLOAD_FOLDER'])
    clear_old_files(app.config['OUTPUT_FOLDER'])

    if 'image' not in request.files:
        return jsonify({'result': 'error', 'error': 'No image uploaded.'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'result': 'error', 'error': 'No image selected.'})

    filename = secure_filename(image.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(upload_path)

    output_filename = f"analyzed_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        diagram_base64, analysis_data = analyze_karang(upload_path, output_path)
    except Exception as e:
        return jsonify({'result': 'error', 'error': str(e)})

    detected_image_url = f'http://127.0.0.1:5001/outputs/{output_filename}'

    return jsonify({
        'result': 'success',
        'image_url': detected_image_url,
        'diagram': f'data:image/png;base64,{diagram_base64}',
        'analysis_data': analysis_data
    })

@app.route('/outputs/<filename>')
def get_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

