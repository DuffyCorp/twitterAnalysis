import React, { useEffect, useState } from 'react';
import './hash.scss';
import SentimentDisplay from '../../components/sentimentDisplay/sentimentDisplay';
import EmotionDisplay from '../../components/emotionDisplay/EmotionDisplay';

const Hash = (params) => {

  const data = params.data

  const [showSentiment, setShowSentiment] = useState(true);

  const [radarChartData, setRadarChartData] = useState([]);

  const getRadarData = async (data) => {
    let hashArray = [];
    const counts = {};
    let returnData = [];

    await data.forEach((page) => {
      hashArray = hashArray.concat(page.keywords);
    });

    for (const num of hashArray) {
      counts[num] = counts[num] ? counts[num] + 1 : 1;
    }

    for (const key in counts) {
      returnData.push({
        label: key,
        count: counts[key],
      });
    }

    var sorted = returnData.sort(({ count: a }, { count: b }) => b - a);

    var returnArray = sorted.splice(0, 6);

    setRadarChartData(returnArray);
  };

  useEffect(() => {
    if (data?.length > 0 || data) {
      if (data?.length > 0 && radarChartData.length === 0) {
        getRadarData(data);
      } else {
        console.log(false);
      }
    }
  }, [data, radarChartData]);

    if (radarChartData.length > 0) {
      console.log(radarChartData)
      return (
          <div className="hashBody">
            <div className="hashTitle">
              <h2>{showSentiment ? 'Sentiment analysis' : 'Emotion analysis'}</h2>
              <button onClick={() => setShowSentiment(!showSentiment)}>
                {showSentiment
                  ? 'Show emotion analysis'
                  : 'Show sentiment analysis'}
              </button>
            </div>

            {showSentiment ? <SentimentDisplay data={data} radar={radarChartData} search={params.search}/> : <EmotionDisplay data={data} radar={radarChartData} search={params.search}/>}
          </div>
      );
    }
};

export default Hash;
