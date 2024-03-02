# commands to run
# export FLASK_APP=flask_test  
# export FLASK_ENV=development
# flask run

from flask import Flask, render_template, Response, session, redirect, url_for
from PIL import Image
import os
import cv2
import datetime as dt
import numpy as np
import tensorflow as tf

app = Flask(__name__, root_path="/Users/trac.k.y/Downloads/chihuahua_muffin/app", static_folder=os.path.join("/Users/trac.k.y/Downloads/chihuahua_muffin/app", 'static'))
app.secret_key = "509d5ad1fb502054034e60c79aa439b3"
print("App Root Path:", app.root_path)
print("Static Folder Path:", app.static_folder)

# # old code: save in desktop folder named "images"
# try:
#     os.mkdir('./images')
# except OSError as error:
#     pass

# uploaded Images folder path
app.config['UPLOAD_FOLDER'] = './static'
# Check if the folder directory exists, if not then create it
if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to capture and save an image
def capture_image():
    vid = cv2.VideoCapture(0) 
    ret, frame = vid.read()
    img_name = f'captured_image_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    #cv2.imwrite(f"./images/{img_name}", frame)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    
    #img_path = f"./images/{img_name}"
    cv2.imwrite(img_path, frame)

    vid.release() 
    cv2.destroyAllWindows()

    session['img_path'] = img_path
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
    model = tf.saved_model.load("/Users/trac.k.y/Downloads/chihuahua_muffin/model/test_model")
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
    #img_name = capture_image()
    #return f"Image saved as {img_name}"

    img_name, img_path = capture_image()

    return render_template("capture.html", img_name=img_name, img_path=img_path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results_page')
def results():
    img_path = session.get('img_path', None)
    res = run_model(img_path)
    return render_template("results.html", img_path=img_path, res=res)

if __name__ == '__main__':
    app.run(debug=True)


