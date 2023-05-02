import React, { useEffect } from 'react';
import useFetch from '../../hooks/useFetch';
import Hash from '../../pages/hash/Hash';
import Loading from '../loading/Loading';

const ResultsDisplay = (params) => {
  console.log(params.SearchData);

  const toggleForm = params.toggleForm;

  const setNewData = params.setNewData;

  const { data, loading, error, reFetch } = useFetch(`/analyses`, {
    request: params.SearchData,
  });

  useEffect(() => {
    if(!loading && params.newData){
      reFetch()
      setNewData(false)
    }
  }, [params.newData])

  useEffect(() => {
    if (loading) {
      toggleForm(true);
    } else {
      toggleForm(false);
    }
  }, [loading, toggleForm]);

  if (loading) {
    return <div><Loading /></div>;
  }
  if (error) {
    return <div>Error: {error}</div>;
  }
  if (data) {
    console.log(data.data);

    const DisplayData = data.data;

    return (
      <div className='results'>
        {<Hash data={DisplayData} search={params.SearchData}/>}
      </div>
    );
  }
};

export default ResultsDisplay;
