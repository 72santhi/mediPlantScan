
import React, { useState, useRef, useEffect } from 'react'
import SelectModel from './SelectModel'; 
import { animateScroll as scroll } from 'react-scroll';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom
//import PlantData from './plant_data.json';

const Home = () => {

  const [selectedImage, setSelectedImage] = useState(null);
  const fileInputRef = useRef(null);
  const resultWrapperRef = useRef(null);
  const [showModal, setShowModal] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  

  const handleImageUpload = (event) => {
    if (event.target.files && event.target.files.length > 0) {
      const imageURL = URL.createObjectURL(event.target.files[0]); // Get the URL of the selected image
      setSelectedImage(imageURL);
      console.log(imageURL);
      console.log("Home.js");
      setPredictionResult(null); // Reset prediction result when a new image is selected
      setShowModal(true);
    }
  };

  const handleCloseModal = (result) => {
    setShowModal(false); // Close the modal
    setPredictionResult(result); // Set the prediction result in Home.js
  };

  useEffect(() => {
    if (predictionResult !== null && resultWrapperRef.current) {
      scroll.scrollTo(resultWrapperRef.current.offsetTop - 50, {
        duration: 500,
        delay: 0,
        smooth: 'easeInOutQuart',
      });
    }
  }, [predictionResult]);

  const openSelectModal = () => {
    if (selectedImage && predictionResult === null) {
      return <SelectModel imageUrl={selectedImage} onCloseModal={handleCloseModal} />;
    }
    return null;
  };

  const handleChangeModel = () => {
    setPredictionResult(null); // Reset prediction result
    setShowModal(true); // Open the overlay again
  };

  const handleClick = () => {
    fileInputRef.current.click();
  };


  return (
    <div className = "home-container">

      <h1 className = "entry-title" itemProp = "headline">Medicinal Plant Detector</h1>
    
      <div className="entry-container" itemProp='text'>

        <div className='box'>
          <img src="/Assests/icon.jpg" alt="Preview" />
          <p>Take a selfie and click Detect Now</p>
          <button type="button" className="upload-button" onClick={handleClick}>
            Upload Image
          </button>
          <input
            type="file"
            ref={fileInputRef}
            className="file-input"
            onChange={handleImageUpload}
            accept="image/*"
            style={{ display: 'none' }}
          />
        </div>

      </div>

      
      {showModal && openSelectModal()}


      {predictionResult !== null && (
        <div ref={resultWrapperRef}>
          <div className="result-container">
            <div className="result-content">
              <h3>Prediction Result:{predictionResult}</h3>
              <img src={selectedImage} alt="Selected" />
              <div className='button-class'>
                <Link to={`/plant/${predictionResult}`} className="info-button" state={{ plantName: predictionResult }}>
                    Know about {predictionResult}
                </Link>
                <span className="or-text">or</span>
                <button type="button" className="info-button" onClick={handleChangeModel}>Change Model</button>
              </div>
            </div>
          </div>
        </div>
      )}
      

      <div className='usage-guide'>
        <h1>Usage Guide</h1>
  
        <div className="guide-points-container">
          <ol>
            <li className="guide-point">
              <span className="point-number">1</span>
              <span className="point-description">Description of guide point 1</span>
            </li>
            <li className="guide-point">
              <span className="point-number">2</span>
              <span className="point-description">Description of guide point 2</span>
            </li>
            <li className="guide-point">
              <span className="point-number">3</span>
              <span className="point-description">Description of guide point 3</span>
            </li>
            <li className="guide-point">
              <span className="point-number">3</span>
              <span className="point-description">Description of guide point 3</span>
            </li>
            <li className="guide-point">
              <span className="point-number">3</span>
              <span className="point-description">Description of guide point 3</span>
            </li>
          </ol>
        </div>
      </div>

      <div className='container-1'>
        <div className = "image-container">
          <img src='/Assests/tulasi1.jpg' alt='tulasi'></img>
        </div>
        <div className="text-container">
          <h2>How accurate is your AI Prediction?</h2>
          <p>
            No artificial intelligence model is perfect, and we know that no
            such model will ever exist. However, we are confident that our
            machine-learning algorithms will surprise you with its accuracy.
          </p>
        </div>
      </div>

      <div className='container-1'>
        <div className="text-container">
          <h2>Plant Identification Made Easy</h2>
          <p>
          India, with a rich heritage of floral diversity, is well-known for its medicinal plant wealth,
          but their identification is one of the major burning issues in Ayurvedic Pharmaceutics.
          And hence, we made it easier with our application.
          </p>
        </div>
        <div className = "image-container">
          <img src='/Assests/img-2.jpg' alt='tulasi'></img>
        </div> 
      </div>

      <div className='container-1'>
        <div className = "image-container">
          <img src='/Assests/img-3.jpg' alt='tulasi'></img>
        </div>
        <div className="text-container">
          <h2>Let AI do the hard work</h2>
          <p>
            Every medicinal plant offers unique therapeutic properties. Find out which plant provides
            what benefits just by scanning.
          </p>
        </div>
      </div>

      

      

    </div>
  )
}

export default Home
