from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pythainlp.tokenize import word_tokenize
import re

# ฟังก์ชันสำหรับการประมวลผลข้อความ
def preprocess_text(text):
    return re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', ' '.join(word_tokenize(str(text or ""))))

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์
def text_to_features(text, w2v_model):
    return np.mean([w2v_model.wv[word] for word in preprocess_text(text).split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)

# โหลดโมเดล Word2Vec และ Random Forest
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# โหลดข้อมูลสำหรับ Word2Vec
df = pd.read_excel('Thai_Sentiment.xlsx', sheet_name='Sheet1')
df['Text'] = df['Text'].fillna('').apply(preprocess_text)
w2v_model = Word2Vec(sentences=[text.split() for text in df['Text']], vector_size=100, window=5, min_count=1, workers=4)

# สร้าง Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_text = request.form.get('user_text')
        if user_text:
            features = text_to_features(user_text, w2v_model).reshape(1, -1)
            prediction = rf_model.predict(features)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
