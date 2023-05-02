import React from 'react';
import './ResultTitle.scss';
import RadarChartDisplay from '../RadarChartDisplay/RadarChartDisplay';

const ResultTitle = (params) => {
  const positiveData = params.positiveData;

  const neutralData = params.neutralData;

  const negativeData = params.negativeData;

  const radarChartData = params.radarData;

  const length = params.length;

  if (
    positiveData.length > neutralData.length &&
    positiveData.length > negativeData.length
  ) {
    return (
      <div className="center">
        <div className="left">
          <h2>Search: {params.search}</h2>
          <h2>Page's analysed: {length}</h2>
          <div className="twitterHeader">
            <h2>Analysis:</h2>
            <h2 className="positive">Positive</h2>
          </div>
        </div>
        <div className="right">
          {radarChartData && <RadarChartDisplay chartData={radarChartData} />}
        </div>
      </div>
    );
  } else if (
    negativeData.length > neutralData.length &&
    negativeData.length > positiveData.length
  ) {
    return (
      <div className="center">
        <div className="left">
          <h2>Search: {params.search}</h2>
          <h2>Page's analysed: {length}</h2>
          <div className="twitterHeader">
            <h2>Analysis:</h2>
            <h2 className="negative">Negative</h2>
          </div>
        </div>
        <div className="right">
          {radarChartData && <RadarChartDisplay chartData={radarChartData} />}
        </div>
      </div>
    );
  } else {
    return (
      <div className="center">
        <div className="left">
          <h2>Search: {params.search}</h2>
          <h2>Page's analysed: {length}</h2>
          <div className="twitterHeader">
            <h2>Analysis:</h2>
            <h2 className="neutral">Neutral</h2>
          </div>
        </div>
        <div className="right">
          {radarChartData && <RadarChartDisplay chartData={radarChartData} />}
        </div>
      </div>
    );
  }
};

export default ResultTitle;
