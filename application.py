from tensorflow.keras.models import load_model
import cv2
import numpy as np

import os
from flask import Flask, render_template, request, redirect, url_for, send_file


def read_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (64, 64))
    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_array




def generate_image(path):
  input_image = read_image(path)
  # input_image.shape


  predicted_mask = model.predict(np.expand_dims(input_image, axis=0))
  predicted_mask = predicted_mask[0, :, :, 0]

  threshold = 0.5 
  binary_mask = (predicted_mask > threshold).astype(np.uint8)
  binary_mask = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]))

  foreground = cv2.bitwise_and(input_image, input_image, mask=binary_mask)
  background = np.ones_like(input_image) * 0  # White background

  result = cv2.add(foreground, background)

  cv2.imwrite(f"NoBackImages/output_image.jpg", result)



app = Flask(__name__)
model = load_model('back_removal.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('error.html')
    
    image = request.files['image']
    
    if image.filename == '':
        return render_template('error.html')
    

    image_path = os.path.join('uploads', 'input_image.jpg')
    image.save(image_path)

    generate_image(image_path)

    return render_template('imageDisplay.html', image_filename='output_image.jpg')
    # return render_template('index.html')



# os.makedirs('uploads', exist_ok=True)
app.run(debug=True, port=5001)

