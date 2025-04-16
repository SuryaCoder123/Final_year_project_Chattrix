from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def is_abusive(text):
    if not text or not text.strip():
        return False
        
    score = analyzer.polarity_scores(text)
    # More nuanced detection:
    # - Highly negative sentiment
    # - Excessive negativity in individual words
    # - Mixed with high intensity
    abusive_words = ['hate', 'stupid', 'idiot', 'kill']  # Add more to this list
    contains_abusive_word = any(word in text.lower() for word in abusive_words)
    
    return (score['compound'] <= -0.6 or 
            (score['neg'] > 0.8 and contains_abusive_word) or
            (score['neu'] < 0.5 and score['compound'] <= -0.4))