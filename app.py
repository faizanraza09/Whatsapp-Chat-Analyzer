import streamlit as st
import pandas as pd
from parse_whatsapp import parse_whatsapp_chat
from sentiment_analysis import add_sentiment_scores, plot_sentiment_over_time
from visualizations import (
    plot_message_count_over_time, 
    get_user_activity_stats,
    plot_user_activity, 
    plot_hourly_activity_heatmap,
    calculate_response_times,
    plot_response_times,
    generate_word_cloud,
    top_emojis_per_user,
    prepare_message_length_data,
    plot_message_length_usage
)
from chatbot import extract_details, filter_chat_by_details

# Set wide mode
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize the Streamlit app
st.title("WhatsApp Chat Analyzer")

# File uploader widget
uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type="txt")
st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space

if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    
    # Parse the chat data
    chat_df = parse_whatsapp_chat(file_content)
    
    st.subheader("Ask Anything about the chat")
    user_query = st.text_input("Enter your query:")

    if st.button("Ask your query"):
        dates, people, topic = extract_details(user_query)
        st.dataframe(filter_chat_by_details(chat_df, people, dates, topic), use_container_width=True)

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    
    # Display the parsed chat data
    st.subheader("Parsed Chat Data")
    st.dataframe(chat_df.sort_values(by='Datetime',ascending=False).reset_index(drop=True), use_container_width=True)

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    st.subheader("Message Count Analysis")

    min_date = chat_df['Datetime'].min().to_pydatetime()
    max_date = chat_df['Datetime'].max().to_pydatetime()

    start_date2, end_date2 = st.slider("Select Date Range for Message Count Graph:", value=(min_date, max_date), format="MM/DD/YY")
    start_date2 = pd.to_datetime(start_date2)  
    end_date2 = pd.to_datetime(end_date2)

    if st.button('Show Message Count Over Time Graph'):
        st.pyplot(plot_message_count_over_time(chat_df,start_date2,end_date2))

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    
    # User Activity Table
    st.subheader("User Activity and Message Length Analysis")
    # User selection
    all_users = chat_df['User'].unique()
    selected_users = st.multiselect('Select Users:', options=all_users, default=all_users)
    
    if selected_users:
        # Filter DataFrame based on selected users
        filtered_chat_df = chat_df[chat_df['User'].isin(selected_users)]

        message_length_df = prepare_message_length_data(filtered_chat_df)
        
        user_activity_df = get_user_activity_stats(message_length_df)
        st.dataframe(user_activity_df, use_container_width=True)

        st.write("<br><br>", unsafe_allow_html=True)  # Adding space

        # User Activity Plot
        st.pyplot(plot_user_activity(filtered_chat_df))

        
        
        # Message Length Analysis

        st.write("<br><br>", unsafe_allow_html=True)  # Adding space
        st.pyplot(plot_message_length_usage(message_length_df))

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    
    # Display top emojis used by each user
    st.subheader("Top Emojis Used by Each User")
    top_emojis = top_emojis_per_user(chat_df)
    st.dataframe(top_emojis,use_container_width=True)
    
    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    st.subheader("Hourly Activity Heatmap")
    st.pyplot(plot_hourly_activity_heatmap(chat_df))

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    st.subheader("Response Time Distribution")
    st.pyplot(plot_response_times(calculate_response_times(chat_df)))

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    st.subheader("Sentiment Analysis Over Time")
    # Add sentiment scores
    chat_df = add_sentiment_scores(chat_df)
    # Date range selector
    start_date, end_date = st.slider("Select Date Range for Sentiment Analysis:", value=(min_date, max_date), format="MM/DD/YY")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if st.button('Show Sentiment Analysis'):
        st.pyplot(plot_sentiment_over_time(chat_df, start_date, end_date))

    st.write("<br><br><br><br>", unsafe_allow_html=True)  # Adding space
    # Generate and display the word cloud
    st.subheader("Chat Word Cloud")
    st.pyplot(generate_word_cloud(chat_df))
