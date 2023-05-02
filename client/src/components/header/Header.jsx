import React, { useState } from 'react';
import './header.scss';

const Header = () => {
  const [toggle, toggleNav] = useState(false);

  return (
    <div className="navContainer">
      <nav className="navBar">
        <a className="menuLink" href="/">
          <h1 className="Logo">Twitter Sentiment</h1>
        </a>
        <ul className="Menu">
          <li>
            <a className="menuLink" href="/">
              Home
            </a>
          </li>
          <li>
            <a className="menuLink" href="/about">
              About
            </a>
          </li>
          {/* <li>
            <a className='menuLink'>
              Github
            </a>
          </li> */}
        </ul>
        <button className="navIcon" onClick={() => toggleNav(!toggle)}>
          <span className="Line" open={toggle} />
          <span className={toggle ? 'LineOpen' : 'Line'} open={toggle} />
          <span className="Line" open={toggle} />
        </button>
      </nav>
      <div className={toggle ? 'OverlayOpen' : 'Overlay'} open={toggle}>
        <ul
          className={toggle ? 'OverlayMenuOpen' : 'OverlayMenu'}
          open={toggle}
        >
          <li>
            <a className="menuLink" target="#" href="/">
              Home
            </a>
          </li>
          <li>
            <a className="menuLink" target="#" href="/about">
              About
            </a>
          </li>
          {/* <li>
            <a className='menuLink' target="#" href="https://github.com/Igor178">
              Github
            </a>
          </li> */}
        </ul>
      </div>
    </div>
  );
};

export default Header;
