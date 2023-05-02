import React, { useState } from 'react'
import "./ReadMoreText.scss"

const ReadMoreText = (params) => {
    const [showFullText, setShowFullText] = useState(false)

    const text = params.text

    const toggleText = () => {
        setShowFullText(!showFullText)
    }
  return (
    <span>{showFullText? <span><p>{text}</p><p onClick={toggleText} className='readButton'>Read less</p></span> : <span><p className='hidden'>{text}</p><p onClick={toggleText} className='readButton'>Read more</p></span>}</span>
  )
}

export default ReadMoreText