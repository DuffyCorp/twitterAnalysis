import React from 'react';
import './emotionTitle.scss';
import RadarChartDisplay from '../RadarChartDisplay/RadarChartDisplay';

const EmotionTitle = (params) => {
  const sadness = params.sadness;

  const joy = params.joy;

  const fear = params.fear;

  const anger = params.anger;

  const love = params.love;

  const surprise = params.surprise;

  const radarChartData = params.radarData;

  const length = params.length;

  const values = {
    sadness: sadness.length,
    joy: joy.length,
    fear: fear.length,
    anger: anger.length,
    love: love.length,
    surprise: surprise.length,
  };

  // eslint-disable-next-line no-undef
  var maxKey = Object.keys(values).reduce((a, b) => values[a] > values[b] ? a : b);

  return (<div className="center">
  <div className="left">
    <h2>Search: {params.search}</h2>
    <h2>Page's analysed: {length}</h2>
    <div className="twitterHeader">
      <h2>Analysis:</h2>
      <h2 className={maxKey}>{maxKey}</h2>
    </div>
  </div>
  <div className="right">
    {radarChartData && <RadarChartDisplay chartData={radarChartData} />}
  </div>
</div>);
};

export default EmotionTitle;
