from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import os
from metrics import top_2_accuracy,top_3_accuracy

# Create a Flask app instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    # print("Hi")
    # Get the uploaded file from the request object
    file = request.files['image']
    if not file:
        return 'No file uploaded.'
    
    test_image = Image.open(file.stream)

    # test_image = tf.keras.utils.load_img('Brain-MRI-Classification/Brain-MRI/predict//'+
    #                                      os.listdir("Brain-MRI-Classification/Brain-MRI/predict")[i], 
    #                                                 target_size = (150, 150))
    print(test_image)
    test_image = test_image.resize((224,224))
    loaded_model1 = keras.models.load_model('full_skin_cancer_model.h5')
    #loaded_model2 = keras.models.load_model('Skin_Cancer.h5')
    
    test_image= tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    print("Hi ",test_image)
    result1 = loaded_model1.predict(test_image)
    #result2 = loaded_model2.predict(test_image2)
    

    class_idx1 = np.argmax(result1, axis=1)[0]  # get the index of the predicted class
    #class_idx2 = np.argmax(result2, axis=1)[0]
#   print(class_idx)
    class_labels = ['akiec',  'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # define your class labels
    class_label1 = class_labels[class_idx1]
    #class_label2 = class_labels[class_idx2]
    print(result1,sep="\n")
    
    print('Predicted class according to Mobilenet CNN:', class_label1)
    #print('Predicted class according to CNN:', class_label2)

   
    # Load the saved model
    
    
    # Render the prediction result using a template
    return render_template('result.html', tumor_type1=class_label1)

# Run the Flask app
if __name__ == '__main__':
    app.run()