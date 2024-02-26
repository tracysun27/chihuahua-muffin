# from flask import Flask, render_template
# import cv2

# # commands to run
# # export FLASK_APP=flask_test  
# # export FLASK_ENV=development
# # flask run

# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template("home.html")

from flask import Flask, render_template, Response
import cv2
import datetime as dt

app = Flask(__name__)

# Function to capture and save an image
def capture_image():
    vid = cv2.VideoCapture(0) 
    ret, frame = vid.read()
    img_name = f'captured_image_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
    cv2.imwrite(img_name, frame)
    vid.release() 
    cv2.destroyAllWindows()
    return img_name

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

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/capture', methods=['POST'])
def capture():
    img_name = capture_image()
    return f"Image saved as {img_name}"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/results_page')
def results():
    return render_template("results.html")
  
if __name__ == '__main__':
    app.run(debug=True)


