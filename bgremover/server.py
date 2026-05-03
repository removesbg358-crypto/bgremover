from flask import Flask, request, send_file, render_template
from rembg import remove, new_session
import cv2
import numpy as np
import os

app = Flask(__name__)

# ✅ AUTO DOWNLOAD MODEL (IMPORTANT FIX)
session = new_session("isnet-general-use")


def refine_edges(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]

        _, thresh = cv2.threshold(alpha, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(alpha)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            alpha = cv2.bitwise_and(alpha, mask)

        kernel = np.ones((2,2), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3,3), 0)

        img[:, :, 3] = alpha

    _, output = cv2.imencode(".png", img)
    return output.tobytes()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['image']
    input_data = file.read()

    raw_output = remove(input_data, session=session)
    final_output = refine_edges(raw_output)

    output_path = "output.png"
    with open(output_path, "wb") as f:
        f.write(final_output)

    return send_file(output_path, mimetype='image/png')


if __name__ == "__main__":
    app.run()