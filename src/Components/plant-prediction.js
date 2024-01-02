// plant-prediction.js
import { useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import labels from './Labels';

const predictPlant = async (imageUrl, modelName, setResult) => {
  // Load the image and handle prediction
  const imageElement = new Image();
  imageElement.src = imageUrl;
  imageElement.crossOrigin = 'anonymous'; // Enable cross-origin resource sharing (CORS)
  await imageElement.decode();

  const resizedImage = await loadAndResizeImage(imageElement, [299, 299]);
  const tensor = await convertImageToTensor(resizedImage);

  // Load the model if not already loaded
  const model = await loadModel(modelName);

  // Make prediction
  const predictions = await model.predict(tensor);

  // Get the index of the highest prediction score
  const predictedLabelIndex = tf.argMax(predictions.flatten()).dataSync()[0];

  // Get the predicted label based on the index
  const predictedLabel = labels[predictedLabelIndex];

  // Get the confidence score of the prediction
  const confidence = predictions.dataSync()[predictedLabelIndex] * 100;

  // Update the state with the prediction result
  setResult(`This image most likely belongs to ${predictedLabel} with a ${confidence.toFixed(2)} percent confidence.`);
};

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

const loadModel = async (modelName) => {
  // Adjust the model path based on the selected model
  /*const modelPath = modelName === "InceptionV3"
    ? "mediscan/src/Components/InceptionV3.h5"
    : "path_to_other_model.h5";*/

    try {
        const model = await tf.loadGraphModel(`./${InceptionV3.h5}`);
        console.log("Model loaded successfully:", model);
    } catch (error) {
        console.error("Error loading model:", error);
        throw error;
    }
};

const PlantPrediction = ({ imageUrl, setResult }) => {
  useEffect(() => {
    if (imageUrl) {
      // Call the predictPlant function with the selected model
      predictPlant(imageUrl, "InceptionV3", setResult);
    }
  }, [imageUrl, setResult]);

  // No need to render anything directly in this component
  return null;
};

export { predictPlant, PlantPrediction };
