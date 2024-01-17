import { useState, useEffect } from 'react';
import { predictPlant } from './plant-prediction.js';

const usePredictPlant = (imageUrl) => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null); // Clear any previous errors

      try {
        const prediction = await predictPlant(imageUrl);
        setResult(prediction);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [imageUrl]);

  return { result, loading, error };
};

export default usePredictPlant;
