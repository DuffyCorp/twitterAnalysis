import React from 'react';
import Tweet from '../Tweet/Tweet';

const AllEmotionDisplay = (params) => {
  const sadnessData = params.sadness;

  const joyData = params.joy;

  const fearData = params.fear;

  const angerData = params.anger;

  const loveData = params.love;

  const surpriseData = params.surprise;

  return (
    <div className="allTweetsWrapper">
      <h2>All Pages</h2>

      <div className="allTweetsContainer">
        {sadnessData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader sad">Sadness Pages</h3>
            {sadnessData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
        {joyData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader joy">Joy Pages</h3>
            {joyData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
        {fearData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader fear">Fear Pages</h3>
            {fearData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
        {angerData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader anger">Anger Pages</h3>
            {angerData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
        {loveData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader love">Love Pages</h3>
            {loveData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
        {surpriseData.length > 0 && (
          <div className="tweetsContainer">
            <h3 className="tweetContainerHeader surprise">Surprise Pages</h3>
            {surpriseData.map((tweet, index) => {
              return <Tweet key={index} tweet={tweet} />;
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default AllEmotionDisplay;
