"""# เลือก Word2Vec, Random Forest, Split(80:20) เพราะได้ผลลัพธ์ดีที่สุด"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from pythainlp.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import TfidfModel as tfidf

# ฟังก์ชันสำหรับการประมวลผลข้อความ
def preprocess_text(text):
    if text is None:
        return ""  # คืนค่าว่างถ้าข้อความเป็น None

    text = str(text)  # ทำให้มั่นใจว่าเป็นสตริง
    words = word_tokenize(text)  # ใช้ pythainlp ในการตัดคำภาษาไทย
    text = ' '.join(words)  # รวมคำที่ตัดแล้วกลับมาเป็นข้อความเดียว
    text = re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', text)  # ลบตัวอักษรพิเศษ
    return text

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์ด้วย Word2Vec
#def extract_features_word2vec(corpus):
    model = tfidf(sentences=[text.split() for text in corpus], vector_size=100, window=5, min_count=1, workers=4)
    features = np.array([
        np.mean([model.wv[word] for word in text.split() if word in model.wv] or [np.zeros(100)], axis=0)
        for text in corpus
    ])
    return features, model

def extract_features_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=1000)  # เลือกจำนวนฟีเจอร์สูงสุด (เช่น 1000)
    features = vectorizer.fit_transform(corpus).toarray()
    return features, vectorizer

# ฟังก์ชันสำหรับการฝึกและบันทึกโมเดล
def train_and_save_model(features, labels, save_path='models/random_forest_tfidf_model.pkl'):
    model = RandomForestClassifier(random_state=42)
    model.fit(features, labels)

    # บันทึกโมเดลด้วย pickle
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {save_path}")
    return model

# ฟังก์ชันสำหรับการทดสอบโมเดล
def test_model(test_texts, w2v_model, loaded_model):
    processed_texts = [preprocess_text(text) for text in test_texts]
    test_features = np.array([
        np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
        for text in processed_texts
    ])
    predictions = loaded_model.predict(test_features)
    return predictions

# ฟังก์ชันหลัก
def main(file_path):
    # โหลดข้อมูลจากไฟล์ Excel
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df['Text'] = df['Text'].fillna('').apply(preprocess_text)  # ประมวลผลข้อความ

    # แบ่งข้อมูลเป็นชุด Train และ Test
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Class'], test_size=0.2, random_state=42)

    # แปลงข้อความเป็นฟีเจอร์ด้วย TF-IDF
    train_features, tfidf_vectorizer = extract_features_tfidf(X_train)

    # ฝึกและบันทึกโมเดล
    model = train_and_save_model(train_features, y_train, save_path='models/random_forest_tfidf_model.pkl')

    # แปลงชุดทดสอบด้วย TF-IDF (ใช้ vectorizer เดียวกัน)
    test_features = tfidf_vectorizer.transform(X_test).toarray()

    # โหลดโมเดลที่บันทึกไว้
    with open('random_forest_tfidf_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # ทดสอบโมเดล
    predictions = loaded_model.predict(test_features)

    # แสดงผลลัพธ์การประเมิน
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # แสดง Classification Report โดยให้ทศนิยม 4 ตำแหน่ง
    print("Classification Report:")
    report = classification_report(y_test, predictions, output_dict=True)

    # แปลงข้อมูลใน classification report เป็น DataFrame
    report_df = pd.DataFrame(report).T

    # จัดรูปแบบให้มีทศนิยม 4 ตำแหน่ง
    report_df = report_df.round(4)

    # แสดงตาราง
    print(report_df)

# เรียกใช้งานฟังก์ชันหลัก
main('data/Thai_Sentiment.xlsx')