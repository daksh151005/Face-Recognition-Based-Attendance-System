from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image
import pandas as pd
import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/register_face', methods=['GET'])
def register_face():
    logging.info("Face registration process started.")
    try:
        # Logic to handle face registration
        return jsonify({"message": "Face registration started!"})
    except Exception as e:
        logging.error(f"Error during face registration: {e}")
        return jsonify({"message": "An error occurred while registering the face."}), 500

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        img_data = data.get('image')
        if not img_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Use OpenCV Haar Cascade to detect faces
        haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(haar_cascade_path)
        faces = detector.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

        recognized_names = []

        # Load known face features from CSV
        if not os.path.exists("data/features_all.csv"):
            return jsonify({'error': 'Face database not found'}), 500

        csv_rd = pd.read_csv("data/features_all.csv", header=None)
        face_name_known_list = []
        face_features_known_list = []
        for i in range(csv_rd.shape[0]):
            face_name_known_list.append(csv_rd.iloc[i][0])
            features = []
            for j in range(1, 129):
                if csv_rd.iloc[i][j] == '':
                    features.append(0.0)
                else:
                    features.append(float(csv_rd.iloc[i][j]))
            face_features_known_list.append(np.array(features))

        def euclidean_distance(f1, f2):
            return np.linalg.norm(f1 - f2)

        # For each detected face, assign "Unknown" as placeholder (since no real feature extraction)
        for (x, y, w, h) in faces:
            # Placeholder feature vector
            face_feature = np.zeros(128)
            distances = [euclidean_distance(face_feature, known_feat) for known_feat in face_features_known_list]
            if distances:
                min_dist = min(distances)
                if min_dist < 0.5:
                    idx = distances.index(min_dist)
                    recognized_names.append(face_name_known_list[idx])
                else:
                    recognized_names.append("Unknown")
            else:
                recognized_names.append("Unknown")

        return jsonify({'recognized_faces': recognized_names})

    except Exception as e:
        logging.error(f"Error in recognize_face: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

if __name__ == '__main__':
    app.run(debug=True)
