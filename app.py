from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model("image_model.h5")

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (32,32))
    img = img/255.0
    img = np.reshape(img,(1,32,32,3))
    return img

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None

    if request.method == 'POST':
        file = request.files['image']
        path = "static/" + file.filename
        file.save(path)

        img = preprocess_image(path)
        pred = model.predict(img)
        prediction = np.argmax(pred)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)