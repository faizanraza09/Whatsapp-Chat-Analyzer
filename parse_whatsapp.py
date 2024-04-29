import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')


def parse_whatsapp_chat(file_content):
    lst = []
    for line in file_content.split('\n'):
        line = line.strip()
        if re.match(r'^\[\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[AP]M\]', line):
            date = re.findall(r'\[(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[AP]M)\]', line)[0]
            user = re.findall(r'\] ([^:]+):', line)
            message = re.findall(r': (.+)$', line)

            if len(user) > 0 and len(message) > 0:
                cleaned_message = clean_and_tokenize(message[0])
                lst.append([date, user[0], message[0], cleaned_message])
            else:
                lst.append([date, user[0], line, []])

    whatsapp_df = pd.DataFrame(lst, columns=['Datetime', 'User', 'Message', 'Tokens'])
    whatsapp_df['Datetime'] = pd.to_datetime(whatsapp_df['Datetime'], format='%m/%d/%y, %I:%M:%S %p')
    return whatsapp_df.sort_values(by='Datetime')

def clean_and_tokenize(text):
    # Lowercase conversion
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens
