from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)
model = None

def load_model():
  global model
  model = ResNet50(weights="imagenet")

def prepare_image(image, target):
  if image.mode != "RGB":
    image = image.convert("RGB")

  image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)


  return image

def dog_detector(data):
  prediction = np.argmax(data)
  return ((prediction <= 268) & (prediction >= 151))

# BELOW IS ONLY INCLUDED FOR GUNICORN
load_model()

@app.route("/predict", methods=["POST"])
def predict():

  data = {"success": False}

  if flask.request.method == "POST":
    if flask.request.files.get("image"):
      # read the image in PIL format
      image = flask.request.files["image"].read()
      image = Image.open(io.BytesIO(image))

      image = prepare_image(image, target=(224, 224))

      preds = model.predict(image)
      results = imagenet_utils.decode_predictions(preds)
      data["predictions"] = []

      if dog_detector(preds):
        data["success"] = True
      else:
        data["success"] = False

      for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data["predictions"].append(r)
  return flask.jsonify(data)

# BELOW IS COMMENTED OUT BECAUSE IT DOESNT WORK WITH GUNICORN
if __name__ == "__main__":
  print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
  load_model()
  app.run('0.0.0.0', debug = False, threaded = False)

# app.run(debug = False, threaded = False)


