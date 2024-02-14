import React from 'react';
import '../App.css';
import { Link } from "react-router-dom";

const Navbar = () => {
    
    return (
    <div className="main-header">
        <div className = "main-header-container">
            <div className = "logo-container">
                <h6 className='logo'>MediScan</h6>
            </div>
            <nav className="navbar-links">
                <Link to="/">Home</Link>
                <Link to="/Search">Search</Link>
                <Link to="/Feedback">Feedback</Link>
                <Link to="/Contact">Contact</Link>
                <Link to="/Account">Account</Link>
            </nav>
        </div>
    </div>
    );
};

export default Navbar;
