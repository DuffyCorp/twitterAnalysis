import { useState } from 'react';
import Layout from '../../components/Layout';
import './home.scss';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch } from '@fortawesome/free-solid-svg-icons';
import ResultsDisplay from '../../components/ResultsDisplay/ResultsDisplay';

const Home = () => {
  const [search, setSearch] = useState('');
  const [showSearch, setShowSearch] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [disableForm, setDisableForm] = useState(false)
  const [newData, setNewData] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (search === '') return;

    if(showSearch !== ''){
      setNewData(true)
    }
    setShowSearch(search)

    setShowResults(true)
    
  };

  const toggleForm = (change) => {
    setDisableForm(change)
  }

  return (
    <Layout showMenus={false}>
      <div id="particle-container">
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
      </div>
      <div className={showResults ? "homeContainer shrink"  : "homeContainer"}>
        <h1 className='twitterHeader'>{"Google Search analysis"}</h1>
        <div className="formContainer" id="particle-container">
          <form onSubmit={(e) => handleSubmit(e)}>
            <fieldset disabled={disableForm}>
            <div className="searchbox js-searchbox">
              <div className="searchbox-icon js-searchbox-icon">
                <FontAwesomeIcon icon={faSearch} onClick={handleSubmit} />
              </div>
              <div className="searchbox-input">
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="input js-input"
                  id="searchInput js-searchInput"
                  placeholder={"Enter Google Search"}
                  autoComplete="off"
                  autoFocus
                />
              </div>
            </div>
            </fieldset>
          </form>
        </div>
      </div>
      {showResults && <ResultsDisplay SearchData={showSearch} toggleForm={toggleForm} newData={newData} setNewData = {setNewData}/>}
    </Layout>
  );
};

export default Home;
