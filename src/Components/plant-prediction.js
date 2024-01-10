// plant-prediction.js
import * as tf from '@tensorflow/tfjs';
import labels from './Labels';

const loadAndResizeImage = async (imageElement, targetSize) => {
  const image = await tf.browser.fromPixels(imageElement);
  const resizedImage = tf.image.resizeBilinear(image, targetSize);
  return resizedImage;
};

const convertImageToTensor = async (image) => {
  const normalized = tf.cast(image, 'float32') / 255.0;
  const expanded = tf.expandDims(normalized, 0);
  return expanded;
};

const loadModel = async () => {
  
  try {
    const model = await tf.loadLayersModel('C:/Users/Santhi_311/Documents/mediPlantScan/src/Components/InceptionV3/model.json');
    console.log("Model loaded successfully:", model);
    return model;
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
};

const predictPlant = async (imageUrl) => {
  // Load the image and handle prediction
  const imageElement = new Image();
  imageElement.src = imageUrl;
  imageElement.crossOrigin = 'anonymous'; // Enable cross-origin resource sharing (CORS)
  await imageElement.decode(); // Ensure the image is loaded

  // Load and resize the image
  const resizedImage = await loadAndResizeImage(imageElement, [299, 299]);
  const tensor = await convertImageToTensor(resizedImage);

  // Load the model if not already loaded
  const model = await loadModel();

  // Make prediction
  const predictions = await model.predict(tensor);

  // Get the index of the highest prediction score
  const predictedLabelIndex = tf.argMax(predictions.flatten()).dataSync()[0];

  // Get the predicted label based on the index
  const predictedLabel = labels[predictedLabelIndex];

  // Get the confidence score of the prediction
  const confidence = predictions.dataSync()[predictedLabelIndex] * 100;

  // Return the prediction result
  return `This image most likely belongs to ${predictedLabel} with a ${confidence.toFixed(2)} percent confidence.`;
};

export { predictPlant };
