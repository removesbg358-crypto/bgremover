from flask import Flask, request, send_file, render_template
from rembg import remove
import os
import uuid

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/remove-bg', methods=['POST'])
def remove_bg():

    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    input_data = file.read()

    # SIMPLE BG REMOVE
    output = remove(input_data)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("/tmp", filename)

    with open(filepath, "wb") as f:
        f.write(output)

    return send_file(filepath, mimetype='image/png')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
