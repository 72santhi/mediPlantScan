import React from "react";
import "./App.css";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Components/Home';
import Contact from './Components/Contact'
import Search from './Components/Search';
import About from './Components/About';
import Account from './Components/Account';
import Navbar from "./Components/Navbar";



function App() {
  return (
    <Router>
      <div className="App">
        <div className="container">
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Search" element={<Search />} />
            <Route path="/About" element={<About />} />
            <Route path="/Contact" element={<Contact />} />
            <Route path="/Account" element={<Account/>} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
