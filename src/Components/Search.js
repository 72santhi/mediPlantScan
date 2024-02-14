import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch } from '@fortawesome/free-solid-svg-icons';
import { useNavigate } from 'react-router-dom';

const SearchBar = ({ plantData }) => {
  const [searchInput, setSearchInput] = useState('');
  const navigate = useNavigate();

  const handleInputChange = (event) => {
    setSearchInput(event.target.value);
  };

  const handleSearch = () => {
    navigate(`/plant/${searchInput}`, { state: { plantData } });
  };

  

  return (
    <div className="search-container">
      <div className='search-heading'>
        <h1>Search for a herb here</h1>
      </div>
      <div className='search-bar'>
        <input
          autoComplete="off"
          spellCheck="false"
          autoCapitalize="none"
          autoCorrect="off"
          id="search_input"
          tabIndex="0"
          className="search-input"
          type="text"
          placeholder="Search for Plants..."
          value={searchInput}
          onChange={handleInputChange}
        />
        <button className="search-icon" onClick={handleSearch}>
          <FontAwesomeIcon icon={faSearch} />
        </button>
      </div>
    </div>
  );
};

export default SearchBar;
