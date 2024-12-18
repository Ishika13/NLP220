import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

file_path = "/Users/ishikakulkarni/Desktop/Studies/NLP220A4/Tweets.csv"
df = pd.read_csv(file_path)

print(df.columns)

# 1: Find the number of unique users
print("Part 1")
num_unique_users = df['name'].nunique()
print(f"Number of unique users: {num_unique_users}")

print("Part 2")
# 2: For each unique user, compute top-5 words from their tweets using TF-IDF
def get_top_5_words_per_user(df):
    user_top_words = []
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    
    for user, user_data in df.groupby('name'):
        user_tweets = user_data['text'].values
        tfidf_matrix = vectorizer.fit_transform(user_tweets)
        top_words = vectorizer.get_feature_names_out()
        
        user_top_words.append({'user': user, 'top_words': ', '.join(top_words)})

    top_words_df = pd.DataFrame(user_top_words)
    
    print(top_words_df.head(5))

    top_words_df.to_csv('/Users/ishikakulkarni/Desktop/Studies/NLP220A4/top_5_words_per_user.csv', index=False)
    print("\nTop words for all users have been saved to 'top_5_words_per_user.csv'.")

top_5_words = get_top_5_words_per_user(df)

print("Part 3")
# 3: For each airline, find the most active users and for each such user, find their tweets, tweeting location, and tweet-sentiment
def most_active_users_by_airline(df):
    airline_active_users = {}

    for airline in df['airline'].unique():
        airline_data = df[df['airline'] == airline]
        user_activity = airline_data['name'].value_counts().head(5)
        active_users = user_activity.index

        active_user_data = {}
        for user in active_users:
            user_data = airline_data[airline_data['name'] == user]
            active_user_data[user] = user_data[['text', 'tweet_location', 'airline_sentiment_gold']]

        airline_active_users[airline] = active_user_data

    return airline_active_users

active_users = most_active_users_by_airline(df)
for airline, users in active_users.items():
    print(f"\nAirline: {airline}")
    for user, user_data in users.items():
        print(f"  User: {user} | Tweets: {user_data[['text']].head(2)}")

print("Part 4")
# 4: Find the number of missing values for tweet-location and user_timezone field, and drop the missing values
missing_values = df[['tweet_location', 'user_timezone']].isnull().sum()
print(f"\nMissing values in 'tweet_location' and 'user_timezone':\n{missing_values}")

df_cleaned = df.dropna(subset=['tweet_location', 'user_timezone'])

print("Part 5")
# 5: Parse tweet_created field as a date
df_cleaned['tweet_created'] = pd.to_datetime(df_cleaned['tweet_created'], errors='coerce')
print(f"\nData types after parsing 'tweet_created': {df_cleaned.dtypes}")

print("Part 6")
# 6: Find the total number of tweets from Philadelphia and find all different spellings of 'Philadelphia'
def find_philadelphia_tweets(df):
    philadelphia_variants = ['philadelphia', 'philly', 'philadelfia', 'phildelphia']
    pattern = '|'.join(philadelphia_variants)
    philadelphia_tweets = df[df['tweet_location'].str.contains(pattern, case=False, na=False)]
    
    # Normalize all the variations to a standard 'Philadelphia' spelling
    philadelphia_tweets['tweet_location'] = philadelphia_tweets['tweet_location'].replace(
        {variant: 'Philadelphia' for variant in philadelphia_variants}, regex=True)
    
    philadelphia_spellings = philadelphia_tweets['tweet_location'].unique()

    return philadelphia_tweets, philadelphia_spellings

philadelphia_tweets, philadelphia_spellings = find_philadelphia_tweets(df_cleaned)
print(f"\nTotal number of tweets from Philadelphia: {len(philadelphia_tweets)}")
print(f"Different spellings of Philadelphia: {philadelphia_spellings}")

print("Part 7")
# 7: Create a subset where airline_sentiment_confidence is above 0.6 and save as a CSV file
df_subset = df_cleaned[df_cleaned['airline_sentiment_confidence'] > 0.6]
df_subset.to_csv('/Users/ishikakulkarni/Desktop/Studies/NLP220A4/cleaned_tweets_above_confidence.csv', index=False)

print("\nSubset of tweets with confidence above 0.6 saved successfully.")
