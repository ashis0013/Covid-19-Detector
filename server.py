from flask import Flask, render_template, request
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
        prediction = predict(img)
        return render_template('result.html',bruh=prediction)
    else:
        return render_template('index.html')
    

if(__name__ == '__main__'):
    app.run()