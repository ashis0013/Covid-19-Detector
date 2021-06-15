from flask import Flask, render_template, request, jsonify
from PIL import Image
from model import *
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        img = Image.open(request.files['bruh']).convert('RGB')
        pred, score = predict(img)
        classes = ['COVID-19', 'Pneumonia', 'Normal']
        return render_template('result.html', content=[classes[pred],score])
    else:
        return render_template('index.html')
    

if(__name__ == '__main__'):
    app.run()