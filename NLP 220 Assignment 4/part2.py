import re
import html
import nltk
import emoji  # Library to handle emoji conversion
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('wordnet')
nltk.download('punkt')

csv_file = "/Users/ishikakulkarni/Desktop/Studies/NLP220A4/Tweets.csv"

# Dataset
relevant_columns = ["tweet_id", "text"]
data = pd.read_csv(csv_file, usecols=relevant_columns)

print("Data preview before cleaning:")
print(data.head(10))

# Dataset cleaning
def clean_text_data(df, text_col):
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
    
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\$\d+(?:\.\d+)?", "", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        text = emoji.demojize(text) 
        text = re.sub(r":[a-z_]+:", "[EMOJI]", text)  
        text = html.unescape(text)
        text = re.sub(r"([!?.,])\1+", r"\1", text)
        text = re.sub(r"\b(?:\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?(?:\s?[APap][Mm])?)\b", "[TIME/DATE]", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\b\d+\b", "[NUMBER]", text)  
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "they're": "they are",
        }
        for contraction, expansion in contractions.items():
            text = re.sub(rf"\b{contraction}\b", expansion, text)

        # Tokenize and lemmatize the text
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        text = " ".join(tokens)
        
        return text.strip()

    df[text_col] = df[text_col].apply(clean_text)

    df = df.drop_duplicates(subset=[text_col])
    
    df = df[df[text_col].str.strip() != ""]
    
    return df

print("Cleaning text data using the custom function:")
data = clean_text_data(data, "text")

print("Data preview after cleaning:")
print(data.head(10))
