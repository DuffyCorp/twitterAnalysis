import React from 'react';
import Header from './header/Header';
import Footer from './footer/Footer';

const Layout = ({ showMenus = true, children }) => {
  return (
    <div>
      {showMenus && <Header />}
      <main>{children}</main>
      {showMenus && (
        <footer>
          <Footer />
        </footer>
      )}
    </div>
  );
};

export default Layout;
