import React, { useEffect, useState } from 'react';
import Layout from '../../components/Layout';
import { useLocation } from 'react-router-dom';
import useFetch from '../../hooks/useFetch';
import SentimentDisplay from '../../components/sentimentDisplay/sentimentDisplay';
import EmotionDisplay from '../../components/emotionDisplay/EmotionDisplay';

const Keyword = () => {
  const location = useLocation();
  const id = location.pathname.split('/')[2];

  const { data, loading, error } = useFetch(`/keywordAnalyses/${id}`);

  const [showSentiment, setShowSentiment] = useState(true);

  const [radarChartData, setRadarChartData] = useState([]);

  const getRadarData = async (data) => {
    let hashArray = [];
    const counts = {};
    let returnData = [];

    await data.forEach((tweet) => {
      hashArray = hashArray.concat(tweet.hashtags);
    });

    console.log('hash', hashArray);

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
    if (data.length > 0 || data?.data) {
      if (data.data.length > 0 && radarChartData.length === 0) {
        getRadarData(data.data);
      }
    }
  }, [data, radarChartData]);

  if (loading) {
    return <Layout>loading...</Layout>;
  } else {
    if (data.status === 'fail') {
      return (
        <Layout>
          Sorry we cant find that hashtag's sentiment, try another?
        </Layout>
      );
    }
    if (data.status === 'success' && radarChartData.length > 0) {
      console.log('radar', radarChartData);
      return (
        <Layout>
          <div className="hashBody">
            <div className="hashTitle">
              <h2>Keyword
                {showSentiment ? ' sentiment analysis' : ' emotion analysis'}
              </h2>
              <button onClick={() => setShowSentiment(!showSentiment)}>
                {showSentiment
                  ? 'Show emotion analysis'
                  : 'Show sentiment analysis'}
              </button>
            </div>

            {showSentiment ? (
              <SentimentDisplay data={data} radar={radarChartData} />
            ) : (
              <EmotionDisplay data={data} radar={radarChartData} />
            )}
          </div>
        </Layout>
      );
    }
  }
};

export default Keyword;
