import React from 'react';


import { useParams } from 'react-router-dom';

const PlantInfo = () => {

  const { plantName } = useParams();
  const plantData = require(`../Components/plant_data.json`);
  const plantArray = plantData.plants;
  const foundPlant = plantArray.find(plant => plant.name === plantName);
  if (foundPlant) {
    console.log(foundPlant.scientificName);
    console.log(foundPlant.medicinalBenefits);
    // Access other properties as needed
  } else {
    console.error('Plant not found:', plantName);
    // Handle the case where the plant is not found
  }
      

  return (
    <div className="plantInfo">
      {foundPlant ? (
        <>
          <h1>{foundPlant.name}</h1>
          <p>Scientific Name: {foundPlant.scientificName}</p>
          <p>Medicinal Benefits: {foundPlant.medicinalBenefits}</p>
        </>
      ) : (
        <p>Loading plant information...</p>
      )}
    </div>
  );
};

export default PlantInfo;
