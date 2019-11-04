from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image
import io
import numpy as np
import torch
import requests
import base64

import sys
sys.path.append(".")

from api.utils import exif_transpose


app = Flask(__name__)

IMAGE_SIZE = 512

PATH_TO_MODEL = "./api/model/gen_model_epoch_59_cpu.pk"

try:
    generator = torch.load(PATH_TO_MODEL)

except FileNotFoundError:
    url = "https://www.dropbox.com/s/h302ei5jctwp4m6/gen_model_epoch_59_cpu.pk?dl=1"
    r = requests.get(url)
    with open(PATH_TO_MODEL, "wb") as f:
        f.write(r.content)
    generator = torch.load(PATH_TO_MODEL)

generator.eval()


# ---------------------------------------


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result_page():
    output_img_io = colorizer(request=request, img_name="img")
    jpg_as_text = base64.b64encode(output_img_io.getvalue()).decode()
    return render_template('result.html', photo=jpg_as_text)


# ---------------------------------------


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Ping!"})


@app.route('/check_image', methods=['POST'])
def check_image():
    if request.files.get("image"):

        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        input_array = np.asarray(image)

        return jsonify({
            "image_shape": input_array.shape,
            "is_grayscale_image": len(input_array.shape) == 2,
        })


@app.route('/colorize', methods=['POST'])
def colorize():
    if request.files.get("image"):
        output_img_io = colorizer(request=request, img_name="image")
        return send_file(output_img_io, mimetype='image/png')

    else:
        response = jsonify({
            "message": "An image file is required"
        })
        response.status_code = 401
        return response


# ---------------------------------------


def colorizer(request, img_name="image"):
    input_img = request.files[img_name].read()
    input_img = Image.open(io.BytesIO(input_img))

    input_img = exif_transpose(input_img)

    input_img_shape = input_img.size
    input_img = input_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    input_img_array = np.asarray(input_img)

    if len(input_img_array.shape) > 2:
        input_img_array = input_img_array[:, :, 0]

    input_img_array_scaled = ((input_img_array / 256) - 0.5) * 2.0

    with torch.no_grad():
        input_tensor = torch.from_numpy(input_img_array_scaled).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        output_tensor = generator(input_tensor)
        output_img_array_scaled = output_tensor.cpu().numpy()[0].transpose(1, 2, 0)

    output_img_array = (((output_img_array_scaled / 2.0 + 0.5) * 256).astype('uint8'))
    output_img = Image.fromarray(output_img_array)

    output_img = output_img.resize(input_img_shape)

    output_img_io = io.BytesIO()
    output_img.save(output_img_io, 'PNG')
    output_img_io.seek(0)

    return output_img_io






if __name__ == '__main__':
    app.run(debug=True)
