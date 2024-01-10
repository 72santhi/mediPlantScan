//SelectModel.js
import React, { useState } from 'react';
import { predictPlant } from './plant-prediction.js';


const SelectModel = ({ imageUrl, closeModal }) => {
  const [predictionResult, setPredictionResult] = useState(null);

  const handleModelSelection = async (modelName) => {
    closeModal(); // Close the modal when a model is selected

    // Call the predictPlant function with the selected model
    const result = await predictPlant(imageUrl);
    
    // Update the state with the prediction result
    setPredictionResult(result);
  };

  return (
    <div className="SelectModel-overlay" onClick={closeModal}>
      <div className="ModelSelection">

        <div className="left-container">
          {imageUrl && <img src={imageUrl} alt="Selected" />}
        </div>

        <div className="right-container">
          <span className="close" onClick={closeModal}>
            &times;
          </span>
          <h1>Select a Model</h1>
          <h1>Select a Model</h1>
          {predictionResult ? (
            // Display prediction result if available
            <div>
              <h3>Prediction Result:</h3>
              <p>{predictionResult}</p>
            </div>
          ) : (
            // Display model buttons if prediction result is not available
            <div className="Model-button-container">
              <button
                type="button"
                className="model-b"
                onClick={() => handleModelSelection()}
              >
                InceptionV3
              </button>
              <button type="button" className="model-b">
                model2
              </button>
              <button type="button" className="model-b">
                model3
              </button>
              <button type="button" className="model-b">
                model4
              </button>
            </div> 
          )}
        </div>
      </div>  
    </div>
  );
};

export default SelectModel;
