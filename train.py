import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


# -----------------------------
# 1. Data Preprocessing
# -----------------------------
def preprocess_data(dfs, text_col='text', label_col='category'):
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=[text_col, label_col])

    def clean_text(text):
        text = ''.join([c for c in text if c not in string.punctuation])
        words = text.lower().split()
        return ' '.join(words)

    df[text_col] = df[text_col].astype(str).apply(clean_text)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col])

    return df, y, label_encoder


# -----------------------------
# 2. Train Models
# -----------------------------
def train_models(df, y, text_col='text'):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df[text_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return tfidf, nb, rf, X_train, X_test, y_train, y_test


# -----------------------------
# 3. Evaluation
# -----------------------------
def evaluate_models(nb, rf, X_test, y_test, label_encoder):
    y_pred_nb = nb.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}\n")

    print("Classification Report (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred_nb)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Naive Bayes)')
    plt.show()


# -----------------------------
# 4. Word Cloud
# -----------------------------
def plot_wordclouds(df, text_col='text', label_col='category'):
    for cat in df[label_col].unique():
        text = ' '.join(df[df[label_col] == cat][text_col])
        wc = WordCloud(width=600, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {cat.capitalize()}')
        plt.show()


# -----------------------------
# 5. Save/Load Model
# -----------------------------
def save_model(nb, tfidf, label_encoder, prefix='email_classifier'):
    joblib.dump(nb, f'{prefix}_nb.joblib')
    joblib.dump(tfidf, f'{prefix}_tfidf.joblib')
    joblib.dump(label_encoder, f'{prefix}_label_encoder.joblib')


# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    # Example: load datasets (replace with your files)
     # must have 'text' and 'category' columns
    df1 = pd.read_csv("emails_expanded.csv")

    dfs = [df1]

    df, y, label_encoder = preprocess_data(dfs, text_col='text', label_col='category')
    tfidf, nb, rf, X_train, X_test, y_train, y_test = train_models(df, y)

    evaluate_models(nb, rf, X_test, y_test, label_encoder)
    plot_wordclouds(df, text_col='text', label_col='category')

    save_model(nb, tfidf, label_encoder)
    print("✅ Model training complete and saved!")
