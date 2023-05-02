import React from 'react';
import ReadMoreText from '../ReadMoreText/ReadMoreText';

const MostEmotionDataDisplay = (params) => {
  const sadnessData = params.sadness;

  const joyData = params.joy;

  const fearData = params.fear;

  const angerData = params.anger;

  const loveData = params.love;

  const surpriseData = params.surprise;

  const maxSadness = sadnessData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  const maxJoy = joyData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  const maxFear = fearData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  const maxAnger = angerData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  const maxLove = loveData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  const maxSurprise = surpriseData?.reduce(function (prev, current) {
    return prev?.emotion?.score > current?.emotion?.score ? prev : current;
  }, 0);

  return (<div className="mostDataContainer">
  {sadnessData.length > 0 && <div className="mostDataSad">
    <h2>Most Sad page</h2>
    <b>{maxSadness.domain}</b>
    <ReadMoreText text={maxSadness.text} />
  </div>}
  {joyData.length > 0 && <div className="mostDataJoy">
    <h2>Most Joy page</h2>
    <b>{maxJoy.domain}</b>
    <ReadMoreText text={maxJoy.text} />
  </div>}
  {fearData.length > 0 && <div className="mostDataFear">
    <h2>Most Fear page</h2>
    <b>{maxFear.domain}</b>
    <ReadMoreText text={maxFear.text} />
  </div>}
  {angerData.length > 0 && <div className="mostDataAnger">
    <h2>Most Anger page</h2>
    <b>{maxAnger.domain}</b>
    <ReadMoreText text={maxAnger.text} />
  </div>}
  {loveData.length > 0 && <div className="mostDataLove">
    <h2>Most Love page</h2>
    <b>{maxLove.domain}</b>
    <ReadMoreText text={maxLove.text} />
  </div>}
  {surpriseData.length > 0 &&<div className="mostDataSurprise">
    <h2>Most Surprised page</h2>
    <b>{maxSurprise.domain}</b>
    <ReadMoreText text={maxSurprise.text} />
  </div>}
</div>);
};

export default MostEmotionDataDisplay;
