from flask import Flask, render_template, request, jsonify,redirect
from flask_sqlalchemy import SQLAlchemy


import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import load_model
import numpy as np

from PIL import Image
from io import BytesIO
import pandas as pd

# new code 

model = None
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ammar:marketiq@localhost/newdisease'
db = SQLAlchemy(app)

class Predictor(db.Model):
    __tablename__ = 'information'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(40),  unique=True)
    age = db.Column(db.String(10))
    gender = db.Column(db.String(8))
    prediction = db.Column(db.Boolean())

    def __init__(self, name, age, gender, prediction):
        self.name = name
        self.age = age
        self.gender = gender
        self.prediction = prediction


with app.app_context() as cx:
    print('_________________M_____________________')
    db.create_all()




def load_model():
    #open file with model architecture
    json_file = open('model/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(loaded_model_json)

    #load weights into new model
    model.load_weights("model/model.h5")
    print(model.summary())

def process_image(image):
    #read image
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize and convert to tensor
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["POST","GET"])
def index():
    predictions = {}
    if request.method == "POST":
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']

        # only make predictions after sucessfully receiving the file
        if request.files:
            # try:
            print("INSIDEEEE")

            image = request.files["image"].read()
            image = process_image(image)
            out = model.predict(image)
            # send the predictions to index page
            predictions = {"positive": str(
                np.round(out[0][1], 2)), "negative": str(np.round(out[0][0], 2))}
            if predictions['positive'] == "1.0":
                predictor = Predictor(name, age, gender, True)
            else:
                predictor = Predictor(name, age, gender, False)

            db.session.add(predictor)
            db.session.commit()

    return render_template("index.html",predictions=predictions)

@app.route("/data", methods=['GET'])
def data():
    if request.method == "GET":
        myDataList = []
        all_data = db.session.execute(
        db.select(Predictor).order_by(Predictor.name)).scalars()

        for i in all_data:
            myDataList.append({"name": i.name, "age": i.age,
                                "gender": i.gender, "pneumonia": i.prediction})

        df = pd.DataFrame.from_dict(myDataList).set_index('name')
        df.to_excel('data.xlsx')

    return render_template("index.html", reportData=myDataList)

if __name__ == "__main__":
    load_model()
    app.run(debug = True, threaded = False)

if __name__ == "app":
    load_model()
