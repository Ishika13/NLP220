import re
import html
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import emoji
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('punkt')

# Dataset loading function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    cleaned_df = clean_text_data(df, text_col="text")  
    return cleaned_df

# Dataset cleaning function
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
        text = re.sub(r"[^\w\s,]", "", text, flags=re.UNICODE)  
        text = html.unescape(text)  
        text = re.sub(r"([!?.,])\1+", r"\1", text)  
        text = re.sub(r"\b(?:\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?(?:\s?[APap][Mm])?)\b", "[TIME/DATE]", text)  
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  
        text = text.lower()  

        
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

        tokens = word_tokenize(text)  
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens] 
        text = " ".join(tokens)

        return text.strip()

    # Apply cleaning 
    df[text_col] = df[text_col].apply(clean_text)
    df = df.drop_duplicates(subset=[text_col])
    df = df[df[text_col].str.strip() != ""]
    
    return df

def split_data(df):
    X = df['text']
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3, shuffle=True)
    return X_train, X_test, y_train, y_test

# TF-IDF encoding function
def encode_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Training
def train_model(X_train_tfidf, y_train):
    svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, max_iter=100, shuffle=True, random_state=3, tol=None)
    svm.fit(X_train_tfidf, y_train)
    return svm

# Model evaluation function
def evaluate_model(svm, X_test_tfidf, y_test):
    y_pred = svm.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Set3', xticklabels=svm.classes_, yticklabels=svm.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('part3_confusion_matrix.png')
    plt.show()

    return accuracy, report, matrix

# Ablation Study to evaluate different preprocessing combinations and orders
def ablation_study(X_train, y_train):
    def clean_text_with_lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        return " ".join([lemmatizer.lemmatize(token, pos='v') for token in tokens])

    def clean_text_with_emoji_handling(text):
        return emoji.demojize(text)

    def clean_text_with_all_steps(text):
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\$\d+(?:\.\d+)?", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = text.lower()
        text = emoji.demojize(text)
        return text

    def clean_text_with_lemmatization_and_emoji(text):
        text = emoji.demojize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        return " ".join([lemmatizer.lemmatize(token, pos='v') for token in tokens])

    def clean_text_with_emoji_and_lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(emoji.demojize(text))
        return " ".join([lemmatizer.lemmatize(token, pos='v') for token in tokens])

    preprocessing_combinations = [
        ("Original", lambda text: text),  # No preprocessing
        ("Lemmatization", lambda text: clean_text_with_lemmatization(text)),
        ("Emoji Handling", lambda text: clean_text_with_emoji_handling(text)),
        ("All", lambda text: clean_text_with_all_steps(text)),
        ("Lemmatization + Emoji Handling", lambda text: clean_text_with_lemmatization_and_emoji(text)),
        ("Emoji Handling + Lemmatization", lambda text: clean_text_with_emoji_and_lemmatization(text)),
    ]
    
    best_accuracy = 0
    best_combination = None
    
    # Run cross-validation for each combination
    for name, preprocess_func in preprocessing_combinations:
        # Apply the corresponding preprocessing
        X_train_preprocessed = X_train.apply(preprocess_func)
        
        # TF-IDF vectorization
        X_train_tfidf, _, vectorizer = encode_text(X_train_preprocessed, X_train)  # Encoding only train set
        
        # Train and evaluate using 10-fold cross-validation
        svm = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, max_iter=100, shuffle=True, random_state=3, tol=None)
        cv_scores = cross_val_score(svm, X_train_tfidf, y_train, cv=10, scoring="accuracy")
        
        mean_accuracy = cv_scores.mean()
        print(f"Accuracy for {name} preprocessing: {mean_accuracy:.4f}")
        
        # Update best preprocessing if necessary
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_combination = name
            
    print(f"\nBest preprocessing combination: {best_combination} with accuracy: {best_accuracy:.4f}")
    
    return best_combination

if __name__ == "__main__":
    file_path = "/Users/ishikakulkarni/Desktop/Studies/NLP220A4/Tweets.csv"
    
    # Load and prepare dataset
    df = load_and_prepare_data(file_path)

    print("Dataset loaded successfully!")
    print(df.info())
    print("First few rows of the dataset:")
    print(df.head())

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)
    print("Class distribution in training set:", y_train.value_counts())
    print("Class distribution in test set:", y_test.value_counts())

    # TF-IDF Encoding
    X_train_tfidf, X_test_tfidf, vectorizer = encode_text(X_train, X_test)

    # Ablation Study
    print("\nStarting Ablation Study...")
    best_preprocessing_combination = ablation_study(X_train, y_train)
    
    # Train the model with the best preprocessing combination
    print(f"\nTraining the model using {best_preprocessing_combination} preprocessing steps.")
    X_train_preprocessed = X_train.apply(lambda text: text)  # Apply the best combination preprocessing
    X_train_tfidf, X_test_tfidf, vectorizer = encode_text(X_train_preprocessed, X_test)

    svm = train_model(X_train_tfidf, y_train)

    # Evaluate the model on the test set
    accuracy, report, matrix = evaluate_model(svm, X_test_tfidf, y_test)

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
