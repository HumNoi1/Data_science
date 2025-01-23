import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from pythainlp.tokenize import word_tokenize

# Step 1: Preprocess Text
def preprocess_text(text, language='th'):
    if text is None:
        return ""  # คืนค่าว่างถ้า text เป็น None

    text = str(text)  # ทำให้มั่นใจว่าเป็นสตริง

    if language == 'th':
        # สำหรับภาษาไทย
        text = word_tokenize(text)  # ใช้ pythainlp ในการตัดคำภาษาไทย

        # ลบตัวอักษรพิเศษ
        text = re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', ' '.join(text))
        text = text.lower()

        # ลบ stopwords ภาษาไทย
        stop_words_th = set(['แต่', 'และ', 'ที่', 'จาก', 'เป็น', 'จะ', 'กับ', 'ใน', 'ได้', 'ดังนั้น', 'จากนั้น', 'ซึ่ง', 'นั้น', 'หรือ'])
        words = text.split()
        words = [word for word in words if word not in stop_words_th]
    else:
        # สำหรับภาษาอังกฤษ
        nltk.download('stopwords')
        stop_words_en = set(stopwords.words('english'))

        # ลบ mentions, hashtags, URLs, และตัวอักษรพิเศษ
        text = re.sub(r'@\w+|#\w+|http\S+|[^A-Za-z0-9 ]+', '', text)
        text = text.lower()

        # ลบ stopwords
        words = text.split()
        words = [word for word in words if word not in stop_words_en]

    return ' '.join(words)

# Step 2: Load Dataset
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df['Text'] = df['Text'].fillna('')
    df['processed_text'] = df['Text'].apply(preprocess_text, language='th')
    labels = df['Class']
    return df['processed_text'], labels

# Step 3: Feature Extraction
def extract_features(corpus, method='bow'):
    if method == 'bow':
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(corpus)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(corpus)
    elif method == 'word2vec':
        model = Word2Vec(sentences=[text.split() for text in corpus], vector_size=100, window=5, min_count=1, workers=4)
        features = np.array([np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(100)], axis=0) for text in corpus])
    else:
        raise ValueError("Invalid method: choose 'bow', 'tfidf', or 'word2vec'")
    return features

# Step 4: Train and Evaluate
def train_and_evaluate_ensemble(features, labels, folds=5):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }

    results = []
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    for model_name, model in models.items():
        accuracy = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
        results.append({
            'Model': model_name,
            'Mean Accuracy': round(np.mean(accuracy), 4),
            'Standard Deviation': round(np.std(accuracy), 4)
        })

    return pd.DataFrame(results)

# Step 5: Main Function
def main(file_path):
    texts, labels = load_data(file_path)

    # Feature Extraction
    methods = ['bow', 'tfidf', 'word2vec']
    all_results = []

    for method in methods:
        features = extract_features(texts, method)
        results_df = train_and_evaluate_ensemble(features, labels)
        results_df['Method'] = method
        all_results.append(results_df)

    final_results = pd.concat(all_results, ignore_index=True)
    print(final_results)

# Execute
main('data/Thai_Sentiment.xlsx')