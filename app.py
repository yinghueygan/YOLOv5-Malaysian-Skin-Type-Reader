import argparse
import os
import torch
import cv2
import numpy as np
import tensorflow as tf
import keras.utils as image
import ultralytics

from PIL import Image
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
from subprocess import Popen
from keras.models import load_model
from keras.utils import img_to_array, load_img

app = Flask(__name__)

# skin condition detection model
path='best.pt'
model=torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path, force_reload=True)

class_mapping = {
    0: "acne",
    1: "uneven skin",
    2: "fine lines",
    3: "blackhead",
    4: "uneven skin tones",
    5: "enlarged pores",
    6: "shiny",
    7: "flaky",
    8: "redness",
    9: "pigment"
}

# routes for main page
@app.route("/")
def home():
    return render_template("index.html")


# routes for about page
@app.route('/about')
def about():
    return render_template("about.html")


# routes for terms and conditions page
@app.route('/terms')
def terms():
    return render_template("terms.html")


# routes for tips page
@app.route('/tips')
def tips():
    return render_template("tips.html")


# routes for classify skin type page
@app.route("/classify")
def classify():
    return render_template("classify.html")


# routes for classify skin type and detect skin conditions
@app.route("/submit", methods=['GET', 'POST'])
def predict_img():
    # display error message when non-human or obscure face is uploaded
    skin_condition_str = "Error: Non-human or obscure face. Please upload a human face or clearer image."
    skin_type_str = "Error: Non-human or obscure face. Please upload a human face or clearer image."
    skin_type = "Error: Non-human or obscure face. Please upload a human face or clearer image."

    if request.method == "POST":
        if 'file' in request.files:
            # find uploaded image and video path
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'static/uploads/', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)

            # detect skin conditions
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            if file_extension == 'jpg':
                process = Popen(["python", "yolov5/detect.py", '--source', filepath, "--weights", "best.pt"], shell=True)
                process.wait()

                # display skin conditions that detected by the model
                output = model(filepath)
                detect_result = {}
                for obj in output.xyxy[0]:
                    obj_class_idx = int(obj[-1])
                    obj_class_name = class_mapping.get(obj_class_idx, "Unknown")
                    if obj_class_name not in detect_result:
                        detect_result[obj_class_name] = 0
                    detect_result[obj_class_name] += 1

                # display in string, not in list
                skin_condition_str = "\n".join([f"{key}: {value}, " for key, value in detect_result.items()])

                # classify skin type
                for obj_class_name in set(detect_result):
                    obj_class_count = detect_result[obj_class_name]
                    skin_type = ""
                    if obj_class_name == "acne" and obj_class_count >= 4:
                        skin_type = "Oily"
                        break
                    elif obj_class_name == "acne" and obj_class_count <= 3:
                        skin_type = "Dry"
                        break
                    elif obj_class_name == "redness" and obj_class_count >= 1:
                        skin_type = "Sensitive"
                        break
                    elif "acne" not in set(detect_result):
                        skin_type = "Normal"
                        break
                skin_type_str = "".join(skin_type)

            elif file_extension == 'mp4':
                process = Popen(["python", "yolov5/detect.py", '--source', filepath, "--weights", "best.pt"], shell=True)
                process.communicate()
                process.wait()

        folder_path = 'yolov5/runs/detect'
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        latest_subfolder = max(subfolders, key = lambda x: os.path.getctime(os.path.join(folder_path, x)))
        image_path = folder_path + '/' + latest_subfolder + '/' + f.filename

    return render_template('classify.html', skin_condition_str=skin_condition_str, skin_type_str=skin_type_str, image_path=image_path)


# route to display uploaded image and video which are detected
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'yolov5/runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    print("printing directory: ", directory)
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, filename, environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format."


# routes for skincare product recommendation page
@app.route('/product_dry')
def product_dry():
    return render_template("product_dry.html")

@app.route('/product_oily')
def product_oily():
    return render_template("product_oily.html")

@app.route('/product_normal')
def product_normal():
    return render_template("product_normal.html")

@app.route('/product_sensitive')
def product_sensitive():
    return render_template("product_sensitive.html")


# routes to lively detect skin conditions
# function to get the frames and save the image using key q
@app.route("/classify")
def get_frame():
    camera = cv2.VideoCapture(0)
    while True:
        ret, image = camera.read()
        results = model(image)
        a = np.squeeze(results.render())
        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite("C:\\SkinConditionDetector\\static\\uploads\\hi2.jpg", image)

    camera.release()
    cv2.destroyAllWindows()
    return render_template("classify.html")


# route to start webcam and detect skin conditions
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Type Reader App exposing YOLOv5 models.")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','best.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)
