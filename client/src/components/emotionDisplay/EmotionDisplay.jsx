import React from 'react';
import './EmotionDisplay.scss';
import EmotionTitle from '../EmotionTitle/EmotionTitle';
import EmoitionBarChart from '../EmotionBarChart/EmoitionBarChart';
import AllEmotionDisplay from '../AllEmotionDisplay/AllEmotionDisplay';
import MostEmotionDataDisplay from '../MostEmotionDataDisplay/MostEmotionDataDisplay';

const EmotionDisplay = (params) => {
  const data = params.data;

  const radarChartData = params.radar;

  const pageData = data;

  const sadnessData = pageData.filter((page) => {
    return page.emotion.label === 'sadness';
  });

  const joyData = pageData.filter((page) => {
    return page.emotion.label === 'joy';
  });

  const fearData = pageData.filter((page) => {
    return page.emotion.label === 'fear';
  });

  const angerData = pageData.filter((page) => {
    return page.emotion.label === 'anger';
  });

  const loveData = pageData.filter((page) => {
    return page.emotion.label === 'love';
  });

  const surpriseData = pageData.filter((page) => {
    return page.emotion.label === 'surprise';
  });

  return (
    <div className="userContainer">
      <div className="contentContainer">
        {loveData &&
          joyData &&
          fearData &&
          angerData &&
          loveData &&
          surpriseData && (
            <div>
              <EmotionTitle
                sadness={sadnessData}
                joy={joyData}
                fear={fearData}
                anger={angerData}
                love={loveData}
                surprise={surpriseData}
                search={params.search}
                radarData={radarChartData}
                length={data.length}
              />
              <div className="chartContainer">
                <EmoitionBarChart
                  sadness={sadnessData}
                  joy={joyData}
                  fear={fearData}
                  anger={angerData}
                  love={loveData}
                  surprise={surpriseData}
                />
              </div>
              <MostEmotionDataDisplay
                sadness={sadnessData}
                joy={joyData}
                fear={fearData}
                anger={angerData}
                love={loveData}
                surprise={surpriseData}
              />
              <AllEmotionDisplay
                sadness={sadnessData}
                joy={joyData}
                fear={fearData}
                anger={angerData}
                love={loveData}
                surprise={surpriseData}
              />
            </div>
          )}
      </div>
    </div>
  );
};

export default EmotionDisplay;
