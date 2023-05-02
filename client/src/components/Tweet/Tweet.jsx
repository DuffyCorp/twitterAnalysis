import React from 'react';
import './Tweet.scss';
import ReadMoreText from '../ReadMoreText/ReadMoreText';

const Tweet = (tweet) => {
  return (
    <div className="tweetContainer">
      <b>{tweet.tweet.domain}</b>
      <ReadMoreText text={tweet.tweet.text}/>
    </div>
  );
};

export default Tweet;
