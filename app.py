import os
import uuid
import flask
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import supervision as sv
from flask import Flask, render_template, request
from Count_thing.coin import ObjectDetector_Coins
from Count_thing.mango import ObjectDetector_mango
from Count_thing.steal_beam import ObjectDetector_steal_beam
from Count_thing.button import ObjectDetector_button
from Count_thing.wood  import ObjectDetector_wood

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['mango', 'coin','steal_beam','Wood', 'Button']


# Define the models used in this code
model_1=ObjectDetector_Coins('models/best_mango_2.pt')
model_2= ObjectDetector_mango('models/coin_2.pt')
model_3=ObjectDetector_steal_beam('models/steal_beam.pt')
model_4=ObjectDetector_wood('models/wood.pt')
model_5= ObjectDetector_button('models/count_button.pt')

ALLOWED_EXT = {'jpg', 'JPG', 'jpeg', 'png', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 300))
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'Static/images')
    if not os.path.exists(target_img):
        os.makedirs(target_img)

    # Handle mango model prediction
    if request.method == 'POST' and 'file1' in request.files:
        file = request.files['file1']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(target_img, unique_filename)
            file.save(img_path)
            original_img, annotated_img1, num_detections1 = model_1.predict(img_path)
            img1 = unique_filename

            predictions1 = {
                "num_detections1": num_detections1,
            }
            cal1 = {
                "class1": classes[0]
            }

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(annotated_img1, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Predicted Image\nNumber of {classes[0]} detected: {num_detections1}')

            plot_path1 = os.path.join(target_img, "plot_" + unique_filename)
            plt.savefig(plot_path1)
            plt.close()

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if len(error) == 0:
            return render_template('success.html', img=img1, plot=plot_path1, predictions=predictions1, cal=cal1, os=os)
        else:
            return render_template('index.html', error=error, os=os)

    # Handle coin prediction
    elif request.method == 'POST' and 'file2' in request.files:
        file = request.files['file2']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(target_img, unique_filename)
            file.save(img_path)
            original_img, annotated_img2, num_detections2 = model_2.predict(img_path)
            img2 = unique_filename

            predictions2 = {
                "num_detections2": num_detections2,
            }
            cal2 = {
                "class2": classes[1]
            }

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Predicted Image\nNumber of {classes[1]} detected: {num_detections2}')

            plot_path2 = os.path.join(target_img, "plot_" + unique_filename)
            plt.savefig(plot_path2)
            plt.close()

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if len(error) == 0:
            return render_template('success.html', img=img2, plot=plot_path2, predictions=predictions2, cal=cal2, os=os)
        else:
            return render_template('index.html', error=error, os=os)

        # Handle steal_beam prediction
    elif request.method == 'POST' and 'file3' in request.files:
        file = request.files['file3']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(target_img, unique_filename)
            file.save(img_path)
            original_img, annotated_img2, num_detections3 = model_3.predict(img_path)
            img3 = unique_filename

            predictions3 = {
                "num_detections3": num_detections3,
            }
            cal3 = {
                "class3": classes[2]
            }

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Predicted Image\nNumber of {classes[2]} detected: {num_detections3}')

            plot_path3 = os.path.join(target_img, "plot_" + unique_filename)
            plt.savefig(plot_path3)
            plt.close()

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if len(error) == 0:
            return render_template('success.html', img=img3, plot=plot_path3, predictions=predictions3, cal=cal3, os=os)
        else:
            return render_template('index.html', error=error, os=os)
# model bringle

    elif request.method == 'POST' and 'file4' in request.files:
        file = request.files['file4']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(target_img, unique_filename)
            file.save(img_path)
            original_img, annotated_img2, num_detections4 = model_4.predict(img_path)
            img4 = unique_filename

            predictions4 = {
                "num_detections4": num_detections4,
            }
            cal4= {
                "class4": classes[3]
            }

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Predicted Image\nNumber of {classes[3]} detected: {num_detections4}')

            plot_path4 = os.path.join(target_img, "plot_" + unique_filename)
            plt.savefig(plot_path4)
            plt.close()

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if len(error) == 0:
            return render_template('success.html', img=img4, plot=plot_path4, predictions=predictions4, cal=cal4, os=os)
        else:
            return render_template('index.html', error=error, os=os)

# model_ button
    elif request.method == 'POST' and 'file5' in request.files:
        file = request.files['file5']
        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            img_path = os.path.join(target_img, unique_filename)
            file.save(img_path)
            original_img, annotated_img2, num_detections5 = model_5.predict(img_path)
            img5 = unique_filename

            predictions5 = {
                "num_detections5": num_detections5,
            }
            cal5 = {
                "class5": classes[4]
            }

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(annotated_img2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f'Predicted Image\nNumber of {classes[4]} detected: {num_detections5}')

            plot_path5 = os.path.join(target_img, "plot_" + unique_filename)
            plt.savefig(plot_path5)
            plt.close()

        else:
            error = "Please upload images of jpg, jpeg, and png extension only"

        if len(error) == 0:
            return render_template('success.html', img=img5, plot=plot_path5, predictions=predictions5, cal=cal5, os=os)
        else:
            return render_template('index.html', error=error, os=os)
    else:
        return render_template('index.html', os=os)


if __name__ == "__main__":
    app.run(debug=True)
