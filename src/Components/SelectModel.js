import React from 'react';
import { predictPlant } from './plant-prediction.js';

const SelectModel = ({ imageUrl, onCloseModal}) => {
  //const [predictionResult, setPredictionResult] = useState(null);

  const handleModelSelection = async (model) => {
    const result = await predictPlant(imageUrl, model);
    //setPredictionResult(result);
    // Pass the prediction result to the callback in Home.js
    onCloseModal(result);

  };

  return (
    <div className="SelectModel-overlay">
      <div className="ModelSelection">
        <div className="left-container">
          {imageUrl && <img src={imageUrl} alt="Selected" />}
        </div>
        <div className="right-container">
          <span className="close" onClick={() => onCloseModal(null)}>
            &times;
          </span>
          <h1>Select a Model</h1>
          <div className="Model-button-container">
            <button
              type="button"
              className="model-b"
              onClick={() => {
                handleModelSelection("InceptionV3");
                onCloseModal(null); // Close the modal after handling the model selection
              }}
            >
              InceptionV3
            </button>

            <button
              type="button"
              className="model-b"
              onClick={() => {
                handleModelSelection("VGG16");
                onCloseModal(null); // Close the modal after handling the model selection
              }}
            >
              VGG16
            </button>

            <button
              type="button"
              className="model-b"
              onClick={() => {
                handleModelSelection("Ensembled_model");
                onCloseModal(null); // Close the modal after handling the model selection
              }}
            >
              Ensembled VGG16 & VGG19
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SelectModel;
