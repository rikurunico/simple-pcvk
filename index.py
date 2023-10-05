import cv2
import numpy as np
from keras.models import load_model

# Load Model
emotion_model = load_model("emotion_model.h5")
# Inisialisasi objek deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Inisialisasi objek kamera (0 adalah indeks kamera default)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Periksa apakah kamera telah terbuka dengan sukses
if not cap.isOpened():
    print("Kamera tidak dapat diakses.")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Jika frame tidak terbaca, keluar dari loop
    if not ret:
        print("Tidak dapat membaca frame.")
        break

    # Ubah frame menjadi grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    # Loop melalui setiap wajah yang terdeteksi
    for x, y, w, h in faces:
        # Potong wajah dari frame
        face_roi = gray[y : y + h, x : x + w]

        # Ubah ukuran wajah menjadi 48x48 (sesuai dengan ukuran input model)
        resized_face = cv2.resize(face_roi, (48, 48))

        # Normalisasi piksel
        resized_face = resized_face / 255.0

        # Lakukan prediksi emosi pada wajah yang terdeteksi
        emotion_probabilities = emotion_model.predict(
            np.expand_dims(resized_face, axis=0)
        )[0]

        # Ambil indeks emosi dengan probabilitas tertinggi
        emotion_label = np.argmax(emotion_probabilities)

        # Daftar label emosi
        emotion_labels = [
            "Marah",
            "Jijik",
            "Takut",
            "Bahagia",
            "Sedih",
            "Terkejut",
            "Netral",
        ]

        # Tampilkan emosi yang terdeteksi pada frame
        emotion_text = emotion_labels[emotion_label]
        cv2.putText(
            frame,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        # Gambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan frame dengan wajah yang terdeteksi dan emosi yang terdeteksi
    cv2.imshow("Deteksi Emosi", frame)

    # Jika pengguna menekan tombol 'q', keluar dari loop
    if cv2.waitKey(1) == ord("q"):
        break

# Tutup kamera
cap.release()
# Tutup semua window
cv2.destroyAllWindows()
