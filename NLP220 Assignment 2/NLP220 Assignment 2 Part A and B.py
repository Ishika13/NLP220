# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F6KzHYWaOnTsMeH1mdWyKECtDBUuuLk5
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time

# Load the dataset
data = pd.read_csv('/content/ecommerceDataset.csv')
data.columns = ['Category', 'Description']
data.dropna(inplace=True)

data.head()

X = data['Description']
y = data['Category']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, random_state=42, shuffle=True)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_train = X_train.apply(preprocess_text)
X_val = X_val.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Feature Engineering on Training Data
# Bag of Words (BoW)
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_val_bow = bow_vectorizer.transform(X_val)
X_test_bow = bow_vectorizer.transform(X_test)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# N-grams (Bi-grams)
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_train_ngrams = ngram_vectorizer.fit_transform(X_train)
X_val_ngrams = ngram_vectorizer.transform(X_val)
X_test_ngrams = ngram_vectorizer.transform(X_test)

train_data = pd.DataFrame({'Category': y_train})
val_data = pd.DataFrame({'Category': y_val})
test_data = pd.DataFrame({'Category': y_test})

sns.set(style='whitegrid')

# Plots
def plot_class_distribution(data, title):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Category', data=data, palette='pastel')
    plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

plot_class_distribution(train_data, 'Training Set Class Distribution')
plot_class_distribution(val_data, 'Validation Set Class Distribution')
plot_class_distribution(test_data, 'Test Set Class Distribution')

print('Training set class distribution:\n', train_data['Category'].value_counts())
print('Validation set class distribution:\n', val_data['Category'].value_counts())
print('Test set class distribution:\n', test_data['Category'].value_counts())

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv


results = []

def evaluate_model_with_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name, feature_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    end_time = time.time()
    total_time = end_time - start_time

    results.append({
        'model_name': model_name,
        'feature_name': feature_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'time_taken': total_time,
        'confusion_matrix': cm
    })

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PiYG')
    plt.title(f'Confusion Matrix for {model_name} with {feature_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Print metrics
    print(f"\nModel: {model_name} with {feature_name}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Time Taken: {total_time:.2f} seconds")

# Grid search parameters
param_grid_svm = {
    'C': [0.1, 1, 10],
    'loss': ['hinge', 'squared_hinge'],
}

param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
}

param_grid_tree = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
}

# Features
features = [
    (X_train_bow, y_train, X_val_bow, y_val, X_test_bow, y_test, "Bag of Words"),
    (X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test, "TF-IDF"),
    (X_train_ngrams, y_train, X_val_ngrams, y_val, X_test_ngrams, y_test, "N-grams")
]

# Models
models = {
    "SVM": [LinearSVC(), param_grid_svm],
    "Logistic Regression": [LogisticRegression(max_iter=1000), param_grid_logistic],
    "Decision Tree": [DecisionTreeClassifier(), param_grid_tree]
}

for model_name, model in models.items():
    for (X_train, y_train, X_val, y_val, X_test, y_test, feature_name) in features:
        evaluate_model_with_metrics(model[0], X_train, y_train, X_val, y_val, X_test, y_test, model_name, feature_name)

best_result = max(results, key=lambda x: x['accuracy'])

# Print out the best feature and model combination
print("\nBest Model and Feature Combination:")
print(f"Model: {best_result['model_name']}")
print(f"Feature Set: {best_result['feature_name']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"Macro F1: {best_result['f1_score']:.4f}")
print(f"Time taken: {best_result['time_taken']:.2f} seconds")

# Best feature set based on the best_result
if best_result['feature_name'] == "Bag of Words":
    X_train_best = X_train_bow
    X_val_best = X_val_bow
    X_test_best = X_test_bow
elif best_result['feature_name'] == "TF-IDF":
    X_train_best = X_train_tfidf
    X_val_best = X_val_tfidf
    X_test_best = X_test_tfidf
else:
    X_train_best = X_train_ngrams
    X_val_best = X_val_ngrams
    X_test_best = X_test_ngrams

y_train_best = y_train
y_val_best = y_val
y_test_best = y_test

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import HalvingGridSearchCV

# Function to evaluate models
def grid_search_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test):
    # Use HalvingGridSearchCV for efficient hyperparameter tuning
    grid_search = HalvingGridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best CV accuracy for {model.__class__.__name__}: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nPerformance of {model.__class__.__name__} on Test Set:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Macro F1: {f1:.4f}")

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PiYG')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return best_model

# Assuming you have defined models and parameter grids previously
best_models = {}
for model_name, (model, param_grid) in models.items():
    print(f"\nTuning hyperparameters for {model_name} with {best_result['feature_name']} feature set...")
    best_model = grid_search_and_evaluate(model, param_grid, X_train_best, y_train_best, X_test_best, y_test_best)
    best_models[model_name] = best_model

