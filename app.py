import streamlit as st
import pandas as pd
from parse_whatsapp import parse_whatsapp_chat
from sentimentanalysis import add_sentiment_scores, plot_sentiment_over_time
from visualizations import (
    plot_message_count_over_time, 
    plot_user_activity, 
    plot_hourly_activity_heatmap,
    calculate_response_times,
    plot_response_times,
    extract_emojis,
    plot_interaction_network,
    generate_word_cloud,
    message_type_analysis,
)



def top_emojis_per_user(chat_df):
    chat_df['Emojis'] = chat_df['Message'].apply(extract_emojis)
    emoji_list = chat_df[['User', 'Emojis']].explode('Emojis')
    top_emojis = emoji_list.groupby('User')['Emojis'].value_counts().groupby(level=0).head(3).unstack(fill_value=0)
    return top_emojis

# Initialize the Streamlit app
st.title("WhatsApp Chat Analyzer")

# File uploader widget
uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type="txt")
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    
    # Parse the chat data
    chat_df = parse_whatsapp_chat(file_content)
    
    # Display the parsed chat data
    st.subheader("Parsed Chat Data")
    st.dataframe(chat_df)

    # Add sentiment analysis and plot
    chat_df = add_sentiment_scores(chat_df)
    st.pyplot(plot_sentiment_over_time(chat_df))


    st.subheader("Message Count Over Time")
    st.pyplot(plot_message_count_over_time(chat_df))

    st.subheader("User Activity")
    st.pyplot(plot_user_activity(chat_df))

    st.subheader("Hourly Activity Heatmap")
    st.pyplot(plot_hourly_activity_heatmap(chat_df))

    st.subheader("Response Time Distribution")
    st.pyplot(plot_response_times(calculate_response_times(chat_df)))

    # Display top emojis used by each user
    top_emojis = top_emojis_per_user(chat_df)
    st.subheader("Top Emojis Used by Each User")
    st.dataframe(top_emojis)

    # Plot the interaction network
    st.subheader("User Interaction Network")
    st.pyplot(plot_interaction_network(chat_df))

    # Generate and display the word cloud
    st.subheader("Chat Word Cloud")
    st.pyplot(generate_word_cloud(chat_df))

    # Display message type analysis
    st.subheader("Message Type Distribution")
    st.pyplot(message_type_analysis(chat_df))


