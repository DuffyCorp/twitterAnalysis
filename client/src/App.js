import './App.css';
import About from './pages/about/About';
import Home from './pages/home/Home';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import NoPage from './pages/noPage/NoPage';
import Hash from './pages/hash/Hash';
import Keyword from './pages/keyword/Keyword';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/">
          <Route index element={<Home />} />
          <Route path="about" element={<About />} />
          <Route path="hash/:id" element={<Hash />} />
          <Route path="keyword/:id" element={<Keyword />} />
          <Route path="*" element={<NoPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
