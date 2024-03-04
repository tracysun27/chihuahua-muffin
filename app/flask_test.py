from flask import Flask, render_template, Response, session, redirect, url_for, request
from PIL import Image
import os
import cv2
import datetime as dt
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "509d5ad1fb502054034e60c79aa439b3"

# save images in static folder in app
try:
    os.mkdir('./app/static')
except OSError as error:
    pass

# Function to capture and save an image
def capture_image():
    vid = cv2.VideoCapture(0) 
    ret, frame = vid.read()
    img_name = f'captured_image_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    img_path = f"./app/static/{img_name}"
    cv2.imwrite(img_path, frame)
    vid.release() 
    cv2.destroyAllWindows()
    return img_name, img_path

# function for converting captured image into correct dimensions for model
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize the image to match the input size expected by the model
    #img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img

# Function to generate frames from webcam feed
def generate_frames():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_model(img_path, class_names = ["chihuahua", "muffin"]):
    model = tf.saved_model.load("./model/test_model")
    test1 = preprocess_image(img_path)
    res1 = model(test1)
    index = np.argmax(res1)
    # plt.imshow(tf.keras.utils.load_img(img_path))
    return class_names[index]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/capture', methods=['POST',"GET"])
def capture():
    img_name, img_path = capture_image()
    session["img_path"] = img_path
    return redirect(url_for('results', img_name=img_name, img_path=img_path))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results_page')
def results():
    img_path = session.get('img_path', None)
    res = run_model(img_path)
    img_name = request.args.get('img_name')  
    return render_template("results.html", img_path=img_path, img_name=img_name, res=res)

if __name__ == '__main__':
    app.run(debug=True)


