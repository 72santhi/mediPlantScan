from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO
import base64
from flask import request

app = Flask(__name__)
CORS(app)


# Load the model with custom objects
model = tf.keras.models.load_model('InceptionV3.h5')
print("model loaded.....................................................")

# Load your labels
labels = [
    'Aloevera',
    'Amla',
    'Amruthaballi',
    'Arali',
    'Astma_weed',
    'Badipala',
    'Balloon_Vine',
    'Bamboo',
    'Beans',
    'Betel',
    'Bhrami',
    'Bringaraja',
    'Caricature',
    'Castor',
    'Catharanthus',
    'Chakte',
    'Chilly',
    'Citron lime (herelikai)',
    'Coffee',
    'Common rue(naagdalli)',
    'Coriender',
    'Curry',
    'Doddpathre',
    'Drumstick',
    'Ekka',
    'Eucalyptus',
    'Ganigale',
    'Ganike',
    'Gasagase',
    'Ginger',
    'Globe Amarnath',
    'Guava',
    'Henna',
    'Hibiscus',
    'Honge',
    'Insulin',
    'Jackfruit',
    'Jasmine',
    'Kambajala',
    'Kasambruga',
    'Kohlrabi',
    'Lantana',
    'Lemon',
    'Lemongrass',
    'Malabar_Nut',
    'Malabar_Spinach',
    'Mango',
    'Marigold',
    'Mint',
    'Neem',
    'Nelavembu',
    'Nerale',
    'Nooni',
    'Onion',
    'Padri',
    'Palak(Spinach)',
    'Papaya',
    'Parijatha',
    'Pea',
    'Pepper',
    'Pomoegranate',
    'Pumpkin',
    'Raddish',
    'Rose',
    'Sampige',
    'Sapota',
    'Seethaashoka',
    'Seethapala',
    'Spinach1',
    'Tamarind',
    'Taro',
    'Tecoma',
    'Thumbe',
    'Tomato',
    'Tulsi',
    'Turmeric',
    'ashoka',
    'camphor',
    'kamakasturi',
    'kepala'
  ]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        image = request.files['image'].read()
        img = Image.open(BytesIO(image))
        print(img)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.image.resize(img_array, [299, 299])  # Resize the image if needed
        print(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions[0])

        # Get the predicted class and confidence score
        predicted_class = labels[np.argmax(score)]
        confidence = float(np.max(score) * 100)

        # Return the prediction result as JSON
        result = {
            'prediction': predicted_class,
            'confidence': confidence
        }
        print(result)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
