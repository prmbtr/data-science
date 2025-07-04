from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from werkzeug.utils import secure_filename 
import os
from transformers import pipeline
import spacy
import nltk
import PyPDF2
from PyPDF2 import PdfReader
from google import generativeai as genai
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler,Normalizer
import pandas as pd
import numpy as np


app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/condition'
app.config['UPLOAD_FOLDER'] = 'static/skin'
app.config['UPLOAD_FOLDER'] = 'static/bone'

genai.configure(api_key="Your Api Key")

model = genai.GenerativeModel(model_name="gemini-1.5-flash")


@app.route('/')
def home_page():
    return render_template("home.html")


@app.route('/brain',methods=['GET','POST'])
def identify():
    br=load_model(r"C:\Users\HP\Desktop\Natural Language Processing\models_collect\tumour_classification.h5")
    if request.method=='POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = br.predict(img_array)

        """GET TUMOUR TYPES BY MANUALLY WRITING THE NAMES OF TEST DATA AND STROING THEM IN A LIST"""

        b= ['glioma', 'meningioma', 'notumor', 'pituitary']
        pic=sorted(b)
        pred=pic[np.argmax(prediction)]
        return render_template('image.html', r=pred, filename=filename)
    return render_template('image.html', r=None, filename=None)

@app.route('/pneumonia',methods=["GET","POST"])
def identify_chestxray():
    ch = load_model(r"C:\Users\HP\Desktop\Natural Language Processing\models_collect\chest_pne.h5")
    if request.method=='POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = ch.predict(img_array)

        c= ['NORMAL','PNEUMONIA']
        pic=sorted(c)
        pred=pic[np.argmax(prediction)]
        return render_template('xray.html', r=pred, filename=filename)
    return render_template('xray.html', r=None, filename=None)

@app.route("/med_rep",methods=["GET","POST"])
def read():
    if request.method == "POST":
        doc = request.files['pdf_file']
        reader = PdfReader(doc)
        for page in reader.pages:
            text = page.extract_text()

        report_text= text
        lower_text = report_text.lower()
        prompt = (f"analyse {lower_text} and give a detailed summary of patient's medical condition. If {lower_text} doesnot appear to be a medical report then give this output: 'Please upload a medical report'")
        response = model.generate_content(prompt)
        labels = response.text
        return render_template("report_reader.html", r=labels)
    return render_template("report_reader.html")

@app.route('/bone',methods=['GET','POST'])
def frac():
    fr=load_model(r"C:\Users\HP\Desktop\Natural Language Processing\models_collect\bone_fracture.h5")
    if request.method=='POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = fr.predict(img_array)


        b= ['fractured', 'not fractured']
        pic=sorted(b)
        pred=pic[np.argmax(prediction)]
        return render_template('fracture.html', r=pred, filename=filename)
    return render_template('fracture.html', r=None, filename=None)

if __name__=='__main__':
    app.run(debug=True)
