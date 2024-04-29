import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
import networkx as nx

# Set a consistent color scheme
sns.set_theme(style="whitegrid")
colors = sns.color_palette("pastel", 20) 

def plot_message_count_over_time(chat_df):
    """ Generates a plot for message count over time using Matplotlib. """
    plt.figure(figsize=(10, 5))
    chat_df.groupby(chat_df['Datetime'].dt.date).size().plot(kind='line', color=colors[0])
    plt.title('Message Count Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_user_activity(chat_df):
    """ Generates a bar plot for user activity using Matplotlib. """
    plt.figure(figsize=(10, 5))
    user_activity = chat_df['User'].value_counts()
    sns.barplot(x=user_activity.index, y=user_activity.values, palette=colors)
    plt.title('User Activity')
    plt.xlabel('User')
    plt.ylabel('Message Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def calculate_response_times(chat_df):
    """ Calculate response times between messages for each user. """
    chat_df['Response_time'] = chat_df['Datetime'].diff().shift(-1)
    return chat_df

def plot_response_times(chat_df):
    """ Plot response times. """
    response_times = chat_df['Response_time'].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(response_times.dt.seconds / 60, bins=30, color='green')
    plt.title('Response Time Distribution (in minutes)')
    plt.xlabel('Minutes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

def plot_hourly_activity_heatmap(chat_df):
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


def message_type_analysis(chat_df):
    chat_df['Message_Type'] = chat_df['Message'].apply(lambda x: 'Media' if '<Media omitted>' in x else 'Text')
    message_types = chat_df['Message_Type'].value_counts()
    
    plt.figure(figsize=(8, 4))
    message_types.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['gold', 'lightblue'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Message Type Distribution')
    return plt



def generate_word_cloud(chat_df):
    text = ' '.join(chat_df['Message'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Chat Word Cloud')
    return plt


def extract_emojis(text):
    """Extract all emojis from the given text."""
    return ''.join(char for char in text if char in emoji.EMOJI_DATA)

def top_emojis_per_user(chat_df):
    # Extract emojis for each message and associate them with the user
    chat_df['Emojis'] = chat_df['Message'].apply(extract_emojis)
    
    # Explode the DataFrame so each emoji is in its own row but still associated with the user
    all_emojis = chat_df.explode(list(chat_df['Emojis']))
    all_emojis = all_emojis.dropna(subset=['Emojis'])  # remove rows where there are no emojis

    # Group by user and emoji, then count frequencies
    emoji_counts = all_emojis.groupby(['User', 'Emojis']).size().reset_index(name='Counts')

    # Sort the counts and take the top 3 for each user
    top_emojis = emoji_counts.sort_values(['User', 'Counts'], ascending=[True, False]).groupby('User').head(3)

    return top_emojis

def format_top_emojis(top_emojis):
    # Pivot table to have emojis as columns
    result_table = top_emojis.pivot(index='User', columns=top_emojis.groupby('User').cumcount() + 1, values='Emojis').add_prefix('Top Emoji ')
    result_table = result_table.join(
        top_emojis.pivot(index='User', columns=top_emojis.groupby('User').cumcount() + 1, values='Counts').add_prefix('Count ')
    )
    return result_table.reset_index()

def plot_interaction_network(chat_df):
    """ Plot a network graph of user interactions. """
    plt.figure(figsize=(10, 8))
    G = nx.from_pandas_edgelist(chat_df, source='User', target='Message', create_using=nx.Graph())
    pos = nx.spring_layout(G, k=0.15)
    nx.draw_networkx(G, pos, node_color='cyan', edge_color='magenta', with_labels=True, node_size=2500, font_size=9)
    plt.title('Interaction Network')
    plt.axis('off')
    return plt