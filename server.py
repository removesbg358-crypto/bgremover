from flask import Flask, request, send_file, render_template
from rembg import remove, new_session
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# ✅ LAZY LOAD MODEL (IMPORTANT FIX)
session = None

def get_session():
    global session
    if session is None:
        session = new_session("u2netp")  # lightweight model
    return session


def refine_edges(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        return image_bytes

    if img.shape[2] == 4:
        alpha = img[:, :, 3]

        _, thresh = cv2.threshold(alpha, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(alpha)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            alpha = cv2.bitwise_and(alpha, mask)

        kernel = np.ones((2, 2), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        img[:, :, 3] = alpha

    _, output = cv2.imencode(".png", img)
    return output.tobytes()


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    input_data = file.read()

    # ✅ LAZY LOAD SESSION
    raw_output = remove(input_data, session=get_session())

    final_output = refine_edges(raw_output)

    filename = f"output_{uuid.uuid4().hex}.png"
    filepath = os.path.join("/tmp", filename)

    with open(filepath, "wb") as f:
        f.write(final_output)

    return send_file(filepath, mimetype='image/png')


# ✅ RENDER FIX
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
