import React, { useEffect, useState } from 'react';
import ResultTitle from '../ResultTitle/ResultTitle';
import DisplayBarChart from '../DisplayBarChart/DisplayBarChart';
import RadarChartDisplay from '../RadarChartDisplay/RadarChartDisplay';
import MostDataDisplay from '../MostDataDisplay/MostDataDisplay';
import AllTweetsDisplay from '../AllTweetsDisplay/AllTweetsDisplay';

const SentimentDisplay = (params) => {
  const data = params.data;

  const radarChartData = params.radar

  const pageData = data;

  const positiveData = pageData.filter((page) => {
    return page.sentiment.label === 'POSITIVE';
  });

  const negativeData = pageData.filter((page) => {
    return page.sentiment.label === 'NEGATIVE';
  });

  const neutralData = pageData.filter((page) => {
    return page.sentiment.label === 'NEUTRAL';
  });

  return (
    <div className="userContainer">
      <div className="contentContainer">
        {neutralData && (
          <div>
            <ResultTitle
              positiveData={positiveData}
              negativeData={negativeData}
              neutralData={neutralData}
              search={params.search}
              radarData={radarChartData}
              length={data.length}
            />
            <DisplayBarChart
              positiveData={positiveData}
              negativeData={negativeData}
              neutralData={neutralData}
            />
            <MostDataDisplay
              positiveData={positiveData}
              negativeData={negativeData}
            />
            <AllTweetsDisplay
              positiveData={positiveData}
              negativeData={negativeData}
              neutralData={neutralData}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentDisplay;
