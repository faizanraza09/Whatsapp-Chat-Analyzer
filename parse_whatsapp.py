'''
Name(s): Fatema AlMarzooqi, Faizan Raza, Khadiya Khalid
NetID: faa7626, mr5985, kk4597
Course: Language of Computers CADT-UH 1013EQ
Description: This file takes a whatsapp exported chat file and then outputs a pandas dataframe with the parsed chat data
Date: 21/04/2024
'''

import pandas as pd
import re

def parse_whatsapp_chat(file_content):
    lst = []
    for line in file_content.split('\n'):
        line = line.strip()
        if re.match(r'^\[\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[AP]M\]', line):
            date = re.findall(r'\[(\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[AP]M)\]', line)[0]
            user = re.findall(r'\] ([^:]+):', line)
            message = re.findall(r': (.+)$', line)
            
            if len(user) > 0 and len(message) > 0:
                lst.append([date, user[0], message[0]])
            else:
                lst.append([date, user[0], line])

    whatsapp_df = pd.DataFrame(lst, columns=['Datetime', 'User', 'Message'])
    whatsapp_df['Datetime'] = pd.to_datetime(whatsapp_df['Datetime'], format='%m/%d/%y, %I:%M:%S %p')
    return whatsapp_df.sort_values(by='Datetime')
