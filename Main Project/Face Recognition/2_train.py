import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Tentukan direktori dataset
path_dataset = os.path.join(PROJECT_DIR, "dataset")

# 1. Inisialisasi face recognizer dan detektor wajah (menggunakan file XML di folder lokal)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
if detector.empty():
    raise FileNotFoundError(
        f"File cascade tidak ditemukan atau tidak dapat dibaca: "
        f"{os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')}"
    )

def get_images_and_labels(path):
    # Cari semua file gambar di dalam folder dataset
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))

    face_samples = []
    ids = []
    names_dict = {}
    name_to_id = {}
    current_id = 0

    # Petakan nama folder ke ID angka
    for image_path in image_paths:
        # Ambil nama folder sebagai label nama
        nama_subjek = os.path.basename(os.path.dirname(image_path))
        
        if nama_subjek not in name_to_id:
            name_to_id[nama_subjek] = current_id
            names_dict[current_id] = nama_subjek
            current_id += 1
        
        # Cari ID untuk nama ini
        subjek_id = name_to_id[nama_subjek]

        # Baca gambar dan ubah ke grayscale
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah di dalam foto
        faces = detector.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            face_samples.append(gray[y:y+h, x:x+w])
            ids.append(subjek_id)

    return face_samples, np.array(ids), names_dict

print("[INFO] Sedang memproses data wajah untuk training...")
faces, ids, names_dict = get_images_and_labels(path_dataset)

if len(faces) == 0:
    print(f"[ERROR] Tidak ada wajah yang terdeteksi. Periksa foto di folder dataset: {path_dataset}")
else:
    # Lakukan training model
    recognizer.train(faces, ids)

    # Simpan model ke file training.yml
    recognizer.save(os.path.join(BASE_DIR, "training.yml"))

    # Simpan kamus nama (mapping ID ke nama) menggunakan file teks sederhana
    with open(os.path.join(BASE_DIR, "names.txt"), "w") as f:
        for key, value in names_dict.items():
            f.write(f"{key},{value}\n")

    print(f"\n[SUKSES] Training selesai! Model disimpan sebagai 'training.yml'.")
    print(f"[INFO] Terdaftar {len(names_dict)} orang dalam model.")