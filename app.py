from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = "secret123"

model = load_model("best_mask_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# MAIN AI FUNCTION
def predict_multiple_faces(filepath):
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return None, 0, 0, 0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(60, 60))

    mask_count = 0
    no_mask_count = 0

    for (x, y, w, h) in faces:
        face_crop = img_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (128, 128)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        pred = float(model.predict(face_input, verbose=0)[0][0])

        if pred < 0.5:
            mask_count += 1
            color = (0, 255, 0)
            label = "Mask"
        else:
            no_mask_count += 1
            color = (0, 0, 255)
            label = "No Mask"

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_bgr, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = "static/uploads/result.jpg"
    cv2.imwrite(output_path, img_bgr)

    return output_path, mask_count, no_mask_count


# MAIN ROUTE
@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join("static/uploads", file.filename)
        file.save(path)

        img, mask, no_mask = predict_multiple_faces(path)

        return render_template("result.html",
                               image=img,
                               mask_count=mask,
                               no_mask_count=no_mask)

    return render_template("upload.html")


# MAIN START
if __name__ == "__main__":
    app.run(debug=True)
