import React from 'react';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const EmoitionBarChart = (params) => {
  const sadnessData = params.sadness;

  const joyData = params.joy;

  const fearData = params.fear;

  const angerData = params.anger;

  const loveData = params.love;

  const surpriseData = params.surprise;

  const data = [];

  if(sadnessData.length > 0){
    data.push({
      name: 'Sadness',
      Sadness: sadnessData?.length,
    });
  }

  if(joyData.length> 0){
    data.push({
      name: 'Joy',
      Joy: joyData?.length,
    });
  }

  if(fearData.length> 0){
    data.push({
      name: 'Fear',
      Fear: fearData?.length,
    });
  }

  if(angerData.length > 0){
    data.push({
      name: 'Anger',
      Anger: angerData?.length,
    })
  }

  if(loveData.length > 0){
    data.push({
      name: 'Love',
      Love: loveData?.length,
    });
  }

  if(surpriseData.length > 0){
    data.push({
      name: 'Surprise',
      Surprise: surpriseData?.length,
    });
  }
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart
        data={data}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        {sadnessData.length > 0 && <Bar dataKey="Sadness" fill="#2c345c" />}
        {joyData.length > 0 && <Bar dataKey="Joy" fill="#fbe6a6" />}
        {fearData.length > 0 && <Bar dataKey="Fear" fill="#2f2323" />}
        {angerData.length > 0 && <Bar dataKey="Anger" fill="#dd0055" />}
        {loveData.length > 0 && <Bar dataKey="Love" fill="#E41B17" />}
        {surpriseData.length > 0 && <Bar dataKey="Surprise" fill="#f2bcc2" />}
      </BarChart>
    </ResponsiveContainer>
  );
};

export default EmoitionBarChart;
