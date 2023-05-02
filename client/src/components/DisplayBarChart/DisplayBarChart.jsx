import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const DisplayBarChart = (params) => {

  const positiveData = params.positiveData

    const neutralData = params.neutralData

    const negativeData = params.negativeData

    const data = [
        {
          name: 'Positive',
          Positive: positiveData.length,
        },
        {
          name: 'Neutral',
          Neutral: neutralData.length,
        },
        {
          name: 'Negative',
          Negative: negativeData.length,
        },
      ];

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart
          data={data}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="Positive" fill="#00FF00" />
          <Bar dataKey="Neutral" fill="#808080" />
          <Bar dataKey="Negative" fill="#FF0000" />
        </BarChart>
    </ResponsiveContainer>
        
  )
}

export default DisplayBarChart