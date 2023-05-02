import React from 'react';
import './MostDataDisplay.scss';
import ReadMoreText from '../ReadMoreText/ReadMoreText';

const MostDataDisplay = (params) => {
  const positiveData = params.positiveData;

  const negativeData = params.negativeData;

  const maxPositive = positiveData?.reduce(function (prev, current) {
    return prev?.sentiment?.score > current?.sentiment?.score ? prev : current;
  }, 0);

  const maxNegative = negativeData?.reduce(function (prev, current) {
    return prev?.sentiment?.score < current?.sentiment?.score ? prev : current;
  }, 0);

  return (
    <div className="mostDataContainer">
      {maxPositive && <div className="mostDataPositive">
        <h2>Most positive Page</h2>
        <b>{maxPositive.domain}</b>
        <ReadMoreText text={maxPositive.text}/>
      </div>}
      {maxNegative ? <div className="mostDataNegative">
        <h2>Most negative Page</h2>
        <b>{maxNegative.domain}</b>
        <ReadMoreText text={maxNegative.text}/>
      </div> : <></>}
    </div>
  );
};

export default MostDataDisplay;
