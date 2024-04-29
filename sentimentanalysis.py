from textblob import TextBlob
import matplotlib.pyplot as plt

def add_sentiment_scores(chat_df):
    """ Add sentiment polarity and subjectivity to the DataFrame. """
    chat_df['Polarity'] = chat_df['Message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    chat_df['Subjectivity'] = chat_df['Message'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return chat_df

def plot_sentiment_over_time(chat_df):
    """ Plot sentiment polarity over time. """
    plt.figure(figsize=(10, 5))
    plt.plot(chat_df['Datetime'], chat_df['Polarity'], label='Polarity', color='purple')
    plt.title('Sentiment Polarity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

