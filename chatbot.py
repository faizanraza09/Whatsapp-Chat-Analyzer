import spacy
from dateutil import parser

nlp = spacy.load("en_core_web_lg")

# Define a list of words to exclude from topic detection
exclusion_list = ['text', 'message', 'chat', 'conversation', 'send', 'say', 'told', 'asked']

def extract_details(text):
    doc = nlp(text)
    dates = []
    people = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    topics = [token.text for token in doc if token.pos_ == 'NOUN' and token.text.lower() not in exclusion_list and not any(ent.text == token.text for ent in doc.ents)]

    # Extract and parse dates
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            try:
                # Parse the date into a datetime object
                parsed_date = parser.parse(ent.text)
                dates.append(parsed_date)
            except ValueError:
                continue  # If parsing fails, skip this entity

    return dates, people, topics

def filter_chat_by_details(chat_df, people=None, dates=None, topics=None):
    print(topics)
    filtered_df = chat_df
    if people:
        people_regex = "|".join(people)  # Create regex pattern for all people
        filtered_df = filtered_df[filtered_df['User'].str.contains(people_regex, case=False, na=False)]
    if dates:
        filtered_df = filtered_df[filtered_df['Datetime'].apply(lambda x: any(x.date() == date.date() for date in dates))]
    if topics:
        topic_regex = "|".join(topics)  # Create regex pattern for all topics
        filtered_df = filtered_df[filtered_df['Message'].str.contains(topic_regex, case=False, na=False)]
    
    return filtered_df.reset_index(drop=True)