import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
import networkx as nx
from collections import Counter
import pandas as pd

# Set a consistent color scheme
sns.set_theme(style="whitegrid")
colors = sns.color_palette("pastel", 20) 

def plot_message_count_over_time(chat_df, start_date, end_date):
    """
    Generates a plot for message count over time using Matplotlib, filtered by datetime.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    - start_date: Start date for filtering the data
    - end_date: End date for filtering the data
    """
    # Ensure start_date and end_date are datetime objects if they are not
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data based on the datetime range
    mask = (chat_df['Datetime'] >= start_date) & (chat_df['Datetime'] <= end_date)
    filtered_df = chat_df.loc[mask]
    
    plt.figure(figsize=(10, 5))
    filtered_df.groupby(filtered_df['Datetime'].dt.date).size().plot(kind='line', color=colors[0])
    plt.title('Message Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_user_activity_df(chat_df):
    """
    Returns a DataFrame with user message counts.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - DataFrame with user message counts
    """
    return chat_df['User'].value_counts().reset_index().rename(columns={'count': 'Message Count'}).sort_values('Message Count', ascending=False)


def plot_user_activity(chat_df):
    """
    Generates a bar plot for user activity using Matplotlib.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - Matplotlib plot object
    """
    plt.figure(figsize=(10, 5))
    user_activity = chat_df['User'].value_counts()
    sns.barplot(x=user_activity.index, y=user_activity.values, palette=colors)
    plt.title('Message Count by User')
    plt.xlabel('User')
    plt.ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def calculate_response_times(chat_df):
    """
    Calculate response times between messages for each user.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - DataFrame with response times
    """
    chat_df['Response_time'] = chat_df['Datetime'].diff().shift(-1)
    return chat_df

def plot_response_times(chat_df):
    """
    Plot response times.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - Matplotlib plot object
    """
    response_times = chat_df['Response_time'].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(response_times.dt.seconds / 60, bins=30, color=colors[0])
    plt.title('Response Times')
    plt.xlabel('Minutes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

def plot_hourly_activity_heatmap(chat_df):
    """
    Plot a heatmap of hourly activity.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - Matplotlib plot object
    """
    chat_df['Hour'] = chat_df['Datetime'].dt.hour
    chat_df['Weekday'] = chat_df['Datetime'].dt.day_name()
    pivot_table = chat_df.pivot_table(index='Weekday', columns='Hour', aggfunc='size', fill_value=0)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='d')
    plt.title('Hourly Activity Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    return plt

def generate_word_cloud(chat_df):
    """
    Generate a word cloud based on chat messages.
    
    Parameters:
    - chat_df: DataFrame containing chat data
    
    Returns:
    - Matplotlib plot object
    """
    text = ' '.join(chat_df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def top_emojis_per_user(df):
    """
    Get the top emojis used by each user.
    
    Parameters:
    - df: DataFrame containing chat data
    
    Returns:
    - DataFrame with top emojis per user
    """
    def extract_emojis(text):
        return ''.join([char for char in text if char in emoji.EMOJI_DATA])

    def top_emojis(emojis, top_n=3):
        if not emojis:
            return []
        counts = Counter(emojis)
        return [emoji for emoji, _ in counts.most_common(top_n)]

    # Apply the emoji extraction to each message
    df['emojis'] = df['Message'].apply(extract_emojis)

    # Aggregate all emojis used by each user
    user_emoji_series = df.groupby('User')['emojis'].apply(''.join)

    # Calculate the top 3 emojis for each user
    user_top_emojis = user_emoji_series.apply(top_emojis)

    # Prepare the final dataframe
    result = pd.DataFrame({
        'user': user_top_emojis.index,
        'top_3_emojis': user_top_emojis
    }).reset_index(drop=True)

    return result


# Function to prepare message length data
def prepare_message_length_data(chat_df):
    chat_df['Message Length'] = chat_df['Message'].apply(len)
    return chat_df

# Function to plot message length usage
def plot_message_length_usage(chat_df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=chat_df, x='User', y='Message Length', palette=colors, showfliers=False)
    plt.title('Distribution of Message Lengths by Each User')
    plt.xlabel('User')
    plt.ylabel('Message Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt
