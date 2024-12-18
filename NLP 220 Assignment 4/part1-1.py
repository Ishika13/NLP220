import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Specify the direct path to the CSV file
csv_file = "/Users/ishikakulkarni/Desktop/Studies/NLP220A4/Tweets.csv"

# Load the dataset
relevant_columns = ["tweet_id", "airline_sentiment", "text", "retweet_count", 'negativereason', 'airline', 'name']
data = pd.read_csv(csv_file, usecols=relevant_columns)

# Display basic information
print(data.head(10))
print(data.info())

# Data Visualisation

# Plot the number of tweets for each airline
plt.figure(figsize=(10, 6))
sns.countplot(data["airline"], palette="pastel")
plt.title("Number of Tweets for each Airline")
plt.savefig("overall_airline_tweets.png")

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data["airline_sentiment"], palette="pastel")
plt.title("Sentiment Distribution")
plt.savefig("overall_sentiment_distribution.png")

# Total number of data samples
total_samples = data.shape[0]
print(f"Total number of data samples: {total_samples}")

# Group the data by airline
grouped_airlines = data.groupby("airline")

# Loop over each airline to perform the analysis
for airline, group in grouped_airlines:
    print(f"\nAnalysis for Airline: {airline}")
    
    # (1) Total count of data samples for each airline
    total_samples = group.shape[0]
    print(f"Total number of data samples: {total_samples}")
    
    # (2) For the columns “airline_sentiment” and “negativereason”
    unique_sentiments = group["airline_sentiment"].nunique()
    most_frequent_sentiment = group["airline_sentiment"].mode()[0]
    frequency_sentiment = group["airline_sentiment"].value_counts().get(most_frequent_sentiment, 0)
    
    unique_negativereasons = group["negativereason"].nunique()
    most_frequent_negativereason = group["negativereason"].mode()[0]
    frequency_negativereason = group["negativereason"].value_counts().get(most_frequent_negativereason, 0)

    print(f"Unique values in airline sentiment: {unique_sentiments}")
    print(f"Most frequent value in airline sentiment: {most_frequent_sentiment} (Frequency: {frequency_sentiment})")
    print(f"Unique values in negative reason: {unique_negativereasons}")
    print(f"Most frequent value in negative reason: {most_frequent_negativereason} (Frequency: {frequency_negativereason})")

    # (3) Lengths of the shortest and longest tweet for each airline
    group["tweet_length"] = group["text"].apply(len)
    shortest_tweet_length = group["tweet_length"].min()
    longest_tweet_length = group["tweet_length"].max()
    
    print(f"Shortest tweet length: {shortest_tweet_length}")
    print(f"Longest tweet length: {longest_tweet_length}")
    

    # (4) Plot the tweet length distribution in the form of a histogram
    max_length = (group["tweet_length"].max() + 4) // 5 * 5  # Round up to the nearest multiple of 5
    bins = range(0, max_length + 5, 5)  # Create bins for tweet length

    plt.figure(figsize=(10, 6))
    sns.histplot(group["tweet_length"], bins=bins, kde=False, color='pink')
    plt.xlabel("Tweet Length (number of characters)")
    plt.ylabel("Frequency")
    plt.title(f"Tweet Length Distribution for {airline}")
    plt.grid(True)
    plt.savefig(f"tweet_length_distribution_{airline}.png")
    plt.show()


# Unique values in airline sentiment and negativereason
unique_sentiments = data["airline_sentiment"].unique()
unique_negativereasons = data["negativereason"].unique()

print(f"Overall Unique values in airline sentiment: {unique_sentiments}")
print(f"Overall Unique values in negative reason: {unique_negativereasons}")

# Most frequent value in airline sentiment and negativereason
most_frequent_sentiment = data["airline_sentiment"].mode()[0]
most_frequent_negativereason = data["negativereason"].mode()[0]

print(f"Overall Most frequent value in airline sentiment: {most_frequent_sentiment}")
print(f"Overall Most frequent value in negative reason: {most_frequent_negativereason}")

# Frequency of most frequent value in airline sentiment and negativereason
frequency_sentiment = data["airline_sentiment"].value_counts()[most_frequent_sentiment]
frequency_negativereason = data["negativereason"].value_counts().get(most_frequent_negativereason, 0)

print(f"Overall Frequency of most frequent value in airline sentiment: {frequency_sentiment}")
print(f"Overall Frequency of most frequent value in negative reason: {frequency_negativereason}")

# Shortest and longest tweet length
data["tweet_length"] = data["text"].apply(len)
shortest_tweet_length = data["tweet_length"].min()
longest_tweet_length = data["tweet_length"].max()

# Plot the tweet length distribution
max_length = (data["tweet_length"].max() + 4) // 5 * 5 
bins = range(0, max_length + 5, 5)

plt.figure(figsize=(10, 6))
sns.histplot(data["tweet_length"], bins=bins, kde=False, color='pink')
plt.xlabel("Overall Tweet Length (number of characters)")
plt.ylabel("Frequency")
plt.title("Tweet Length Distribution")
plt.grid(True)
plt.savefig("overall_tweet_length_distribution.png")
plt.show()

# Define custom colors for each sentiment
sentiment_colors = {"positive": "#8bca84", "neutral": "#cccccc", "negative": "#FA8072"}

# Create the FacetGrid to create separate histograms for each airline
g = sns.FacetGrid(data, col="airline", col_wrap=3, height=4, aspect=1.5)

# Map the histplot to each subplot
g.map(sns.histplot, "airline_sentiment", bins=3, kde=False)

# Now, loop over each axis to manually color the bars in each subplot
for ax in g.axes.flat:
    for sentiment, color in sentiment_colors.items():
        # Filter data by sentiment and plot each sentiment in a specific color
        sentiment_data = data[data["airline_sentiment"] == sentiment]
        sns.histplot(sentiment_data, x="airline_sentiment", bins=3, color=color, ax=ax, label=sentiment, kde=False)

        # Set axis labels and titles for the subplots
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Frequency")
        ax.set_title(ax.get_title(), fontsize=12)

# Set axis labels and titles
g.set_axis_labels("Sentiment", "Frequency")
g.set_titles("{col_name}")
g.add_legend(title="Sentiment")

# Adjust layout
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Tweet Sentiment Distribution per Airline")

# Save and show the plot
plt.savefig("overall_tweet_sentiment_distribution_per_airline.png")
plt.show()

# Percentage of tweets in each sentiment category per airline
sentiment_percentage = (
    data.groupby(["airline", "airline_sentiment"])
    .size()
    .groupby(level=0)
    .apply(lambda x: (x / x.sum()) * 100)
    .unstack()
)
print("\nPercentage of tweets in each sentiment category per airline:")
print(sentiment_percentage)

# Correlation between retweet count and sentiment
sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
data["sentiment_numeric"] = data["airline_sentiment"].map(sentiment_mapping)
correlation = data[["retweet_count", "sentiment_numeric"]].corr().loc["retweet_count", "sentiment_numeric"]
print(f"\nCorrelation between retweet count and sentiment: {correlation:.4f}")

# Tokenization

def tokenize(text):
    patterns = [
        r"https?://[^\s]+",             
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?",   
        r"\d+(?:\.\d+)?",              
        r"[#@]\w+",                    
        r"[.,!?;:\"'(){}[\]]"
    ]
    
    # Combine patterns into one
    combined_pattern = "|".join(patterns)
    
    # Find all matches using the combined pattern
    tokens = re.findall(combined_pattern, text.strip())
    return tokens

# Tokenize using the above function
print("Using custom tokenization function:")
first_5_tweets = data["text"].head(5)
custom_tokenized_tweets = [tokenize(tweet) for tweet in first_5_tweets]

for i, tokens in enumerate(custom_tokenized_tweets, 1):
    print(f"Tweet {i} tokens: {tokens}")

# NLTK Tokenization
print("Using NLTK tokenization:")
nltk_tokenized_tweets = [word_tokenize(tweet) for tweet in first_5_tweets]
        
for i, tokens in enumerate(nltk_tokenized_tweets, 1):
    print(f"Tweet {i} tokens: {tokens}")

# Identify and print 5 examples where the two tokenizers behave differently
differences = []
for i, (custom, nltk) in enumerate(zip(custom_tokenized_tweets, nltk_tokenized_tweets), 1):
    if custom != nltk:
        differences.append((i, custom, nltk))
        if len(differences) == 5:
            break

print("\nExamples of differences between custom and NLTK tokenizer:")
for diff in differences:
    print(f"Tweet {diff[0]}:")
    print(f"Custom Tokenizer: {diff[1]}")
    print(f"NLTK Tokenizer: {diff[2]}\n")

# Write the differences to a text file
with open("tokenizer_differences.txt", "w") as f:
    f.write("Examples of differences between custom and NLTK tokenizer:\n\n")
    for diff in differences:
        f.write(f"Tweet {diff[0]}:\n")
        f.write(f"Custom Tokenizer: {diff[1]}\n")
        f.write(f"NLTK Tokenizer: {diff[2]}\n\n")
    