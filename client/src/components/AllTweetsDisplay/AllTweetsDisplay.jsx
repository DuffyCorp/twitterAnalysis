import React from 'react';
import './AllTweetsDisplay.scss';
import Tweet from '../Tweet/Tweet';

const AllTweetsDisplay = (params) => {
  const positiveData = params.positiveData;

  const neutralData = params.neutralData;

  const negativeData = params.negativeData;

  return (
    <div className='allTweetsWrapper'>
        <h2>All Pages</h2>
        
      <div className="allTweetsContainer">
        {positiveData.length > 0 && <div className="tweetsContainer">
          <h3 className='tweetContainerHeader positive'>Positive Pages</h3>
          {positiveData.map((tweet, index) => {
            return <Tweet key={index} tweet={tweet} />
        })}
        </div>}
        {neutralData.length > 0 && <div className="tweetsContainer">
          <h3 className='tweetContainerHeader neutral'>Neutral Pages</h3>
          {neutralData.map((tweet, index) => {
            return <Tweet key={index} tweet={tweet} />
        })}
        </div>}
        {negativeData.length > 0 && <div className="tweetsContainer">
          <h3 className='tweetContainerHeader negative'>Negative Pages</h3>
          {negativeData.map((tweet, index) => {
            return <Tweet key={index} tweet={tweet} />
        })}
        </div>}
      </div>
    </div>
  );
};

export default AllTweetsDisplay;
