from flask import Flask, request, render_template_string, send_file
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle
import re
from pythainlp.tokenize import word_tokenize
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import signal

# ฟังก์ชันสำหรับประมวลผลข้อความ
def preprocess_text(text):
    if text is None:
        return ""
    text = str(text)
    words = word_tokenize(text)
    text = ' '.join(words)
    text = re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', text)
    return text

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์ TF-IDF
def extract_features_tfidf(corpus, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(corpus)
    else:
        features = vectorizer.transform(corpus)
    return features, vectorizer

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์ Word2Vec
def extract_features_word2vec(corpus, w2v_model):
    return np.array([np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0) for text in corpus])

# ฟังก์ชันสำหรับบันทึกข้อมูลลงไฟล์
def save_to_file(input_text, predictions):
    with open("analysis_results.txt", "a", encoding="utf-8") as file:
        file.write(f"ข้อความ: {input_text}\n")
        for model_name, prediction in predictions.items():
            file.write(f"{model_name}: {prediction}\n")
        file.write("\n")

# ฟังก์ชันที่หยุดเซิร์ฟเวอร์ Flask
def shutdown_server():
    os.kill(os.getpid(), signal.SIGINT)

# โหลดข้อมูลและโมเดล
df = pd.read_excel('data/Thai_Sentiment.xlsx')
df['Text'] = df['Text'].apply(preprocess_text)

# สร้าง Word2Vec โมเดล
w2v_model = Word2Vec(sentences=[text.split() for text in df['Text']], vector_size=100, window=5, min_count=1, workers=4)

# สร้าง TF-IDF โมเดล
tfidf_features, tfidf_vectorizer = extract_features_tfidf(df['Text'])

# โหลดโมเดลที่บันทึกไว้
with open('models/random_forest_model.pkl', 'rb') as file:
    rf_word2vec = pickle.load(file)

with open('models/random_forest_tfidf_model.pkl', 'rb') as file:
    rf_tfidf = pickle.load(file)

with open('models/gradient_boosting_model.pkl', 'rb') as file:
    gb_word2vec = pickle.load(file)

def plot_bar_chart_per_model(sentiment_counts, model_name):
    fig, ax = plt.subplots(figsize=(10, 7))

    # แยกกราฟสำหรับแต่ละโมเดล
    ax.bar(['Positive', 'Negative', 'Swear'], [
        sentiment_counts[model_name]['positive'],
        sentiment_counts[model_name]['negative'],
        sentiment_counts[model_name]['swear']
    ], color=['blue', 'red', 'green'])

    ax.set_ylabel('จำนวน')
    ax.set_title(f'ผลการวิเคราะห์จากโมเดล: {model_name}')
    ax.set_xticks(['Positive', 'Negative', 'Swear'])

    # Save bar chart to a BytesIO object and return the image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img


def count_sentiment_types(file_path, model_names):
    sentiment_counts = {model: {'positive': 0, 'negative': 0, 'swear': 0} for model in model_names}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        result_line = lines[i + 1]
        for model_name in model_names:
            if model_name in result_line:
                if 'swear' in result_line:
                    sentiment_counts[model_name]['swear'] += 1
                elif 'positive' in result_line:
                    sentiment_counts[model_name]['positive'] += 1
                elif 'negative' in result_line:
                    sentiment_counts[model_name]['negative'] += 1

    return sentiment_counts


# สร้าง Flask App
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
    return render_template_string('''
        <h1>Thai Sentiment Analysis</h1>
        <form method="POST" action="/predict">
            <label>กรอกข้อความภาษาไทย:</label><br>
            <input type="text" name="text" style="width: 300px;"><br><br>
            <input type="submit" value="วิเคราะห์">
        </form>
        <br>
        <a href="/result">ดูผลการวิเคราะห์ทั้งหมด</a><br><br>
        <form method="POST" action="/shutdown">
            <input type="submit" value="ปิดเซิร์ฟเวอร์">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    processed_text = preprocess_text(user_text)

    # แปลงข้อความเป็นฟีเจอร์
    word2vec_features = extract_features_word2vec([processed_text], w2v_model)
    tfidf_features, _ = extract_features_tfidf([processed_text], tfidf_vectorizer)

    # ทำนายผลด้วยโมเดลทั้ง 3
    rf_word2vec_prediction = rf_word2vec.predict(word2vec_features)[0]
    rf_tfidf_prediction = rf_tfidf.predict(tfidf_features)[0]
    gb_word2vec_prediction = gb_word2vec.predict(word2vec_features)[0]

    # รวบรวมผลลัพธ์
    predictions = {
        "Random Forest (Word2Vec)": rf_word2vec_prediction,
        "Random Forest (TF-IDF)": rf_tfidf_prediction,
        "Gradient Boosting (Word2Vec)": gb_word2vec_prediction
    }

    # บันทึกข้อความและผลการวิเคราะห์ลงไฟล์
    save_to_file(user_text, predictions)

    # แสดงผลลัพธ์
    result_html = ''.join([f"<p>{model}: {pred}</p>" for model, pred in predictions.items()])
    return f"<h3>ผลการวิเคราะห์:</h3>{result_html}<br><a href='/'>กลับหน้าหลัก</a>"

@app.route('/result')
def result():
    # นับผลการวิเคราะห์จากไฟล์
    model_names = ["Random Forest (Word2Vec)", "Random Forest (TF-IDF)", "Gradient Boosting (Word2Vec)"]
    sentiment_counts = count_sentiment_types('analysis_results.txt', model_names)

    # สร้างกราฟแท่งสำหรับแต่ละโมเดล
    img_base64_list = []
    for model_name in model_names:
        img = plot_bar_chart_per_model(sentiment_counts, model_name)

        # แปลงภาพกราฟแท่งเป็น base64 สำหรับการแสดงใน HTML
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        img_base64_list.append(img_base64)

    return render_template_string('''
        <h1>ผลการวิเคราะห์ทั้งหมด</h1>
        {% for img_data in img_data_list %}
            <img src="data:image/png;base64,{{ img_data }}" alt="Bar Chart"><br><br>
        {% endfor %}
        <a href="/">กลับหน้าหลัก</a>
    ''', img_data_list=img_base64_list)


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run()