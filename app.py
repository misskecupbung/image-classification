from copyreg import pickle
from flask import Flask, render_template, request
import model
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
  return render_template('index.html')

# Bagian untuk memproses uploaded image
# disimpan ke folder images
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    modelMLP = pickle.load(open("model.pkl", "rb"))
    imgname = image_path
    test = model.imagetoMatrix(imgname)
    label = modelMLP.predict(test)[0]
    classification = '%s' % (label)
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
