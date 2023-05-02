import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';

const RadarChartDisplay = (params) => {
  const chartData = params.chartData;

  return (
    <ResponsiveContainer width={500} height={400}>
      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData} margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}>
        <PolarGrid />
        <PolarAngleAxis dataKey="label" />
        <PolarRadiusAxis />
        <Radar
          name="Mike"
          dataKey="count"
          stroke="#1DA1F2"
          fill="#1DA1F2"
          fillOpacity={0.6}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
};

export default RadarChartDisplay;
