import pickle
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re
import os

class ThaiSentimentAnalyzer:
    """
    คลาสสำหรับวิเคราะห์ความรู้สึกภาษาไทย
    รองรับการใช้งานทั้ง Word2Vec และ TF-IDF โดยควบคุมจำนวน features ให้ตรงกับโมเดล
    """
    def __init__(self, vector_size=100, max_features=854):
        """
        กำหนดค่าเริ่มต้นสำหรับการวิเคราะห์
        
        Parameters:
        vector_size (int): ขนาดของเวกเตอร์สำหรับ Word2Vec
        max_features (int): จำนวน features สูงสุดสำหรับ TF-IDF
        """
        self.vector_size = vector_size
        self.max_features = max_features
        self.models = {}
        self.word2vec_model = None
        self.tfidf_vectorizer = None

    def preprocess_text(self, text):
        """
        เตรียมข้อความภาษาไทยสำหรับการวิเคราะห์
        - ลบอักขระพิเศษ
        - ตัดคำภาษาไทยด้วย PyThaiNLP
        - รวมคำด้วยช่องว่าง
        """
        if pd.isna(text):
            return ""
        # ทำความสะอาดข้อความและตัดคำ
        words = word_tokenize(str(text))
        # ลบอักขระพิเศษและเว้นวรรค
        cleaned_words = [re.sub(r'[^ก-๙a-zA-Z0-9]', '', word) for word in words]
        # กรองคำที่ว่างออก
        cleaned_words = [word for word in cleaned_words if word]
        return ' '.join(cleaned_words)

    def create_word2vec_model(self, texts):
        """
        สร้างโมเดล Word2Vec จากชุดข้อความ
        ควบคุมขนาดของเวกเตอร์ให้ตรงกับที่โมเดลต้องการ
        """
        # เตรียมข้อความสำหรับการฝึกโมเดล
        processed_texts = [
            self.preprocess_text(text).split() 
            for text in texts 
            if pd.notna(text)
        ]
        
        # สร้างและฝึกโมเดล Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=processed_texts,
            vector_size=self.vector_size,  # ใช้ขนาดที่กำหนดในตอนสร้างอินสแตนซ์
            window=5,
            min_count=1,
            workers=4
        )
        print(f"สร้างโมเดล Word2Vec สำเร็จ (vector size: {self.vector_size})")

    def create_tfidf_vectorizer(self, texts):
        """
        สร้างและฝึก TF-IDF Vectorizer
        จำกัดจำนวน features ให้ตรงกับที่โมเดลต้องการ
        """
        # เตรียมข้อความสำหรับ TF-IDF
        processed_texts = [
            self.preprocess_text(text)
            for text in texts
            if pd.notna(text)
        ]
        
        # สร้าง TF-IDF Vectorizer โดยกำหนดพารามิเตอร์ที่เหมาะสม
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,  # จำกัดจำนวน features
            sublinear_tf=True,              # ใช้ logarithmic form สำหรับ term frequency
            strip_accents='unicode',        # ลบเครื่องหมายกำกับเสียง
            analyzer='word',                # วิเคราะห์ระดับคำ
            token_pattern=r'\b\w+\b',       # pattern สำหรับแยกคำ
            ngram_range=(1, 1)             # ใช้เฉพาะ unigrams
        )
        
        # ฝึก vectorizer กับข้อความที่เตรียมไว้
        self.tfidf_vectorizer.fit(processed_texts)
        print(f"สร้าง TF-IDF Vectorizer สำเร็จ (features: {len(self.tfidf_vectorizer.get_feature_names_out())})")

    def extract_features_word2vec(self, text):
        """
        แปลงข้อความเป็นเวกเตอร์โดยใช้ Word2Vec
        คืนค่าเวกเตอร์ที่มีขนาดตรงกับที่โมเดลต้องการ
        """
        if self.word2vec_model is None:
            return np.zeros(self.vector_size)
        
        words = self.preprocess_text(text).split()
        word_vectors = []
        
        for word in words:
            try:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            except AttributeError as e:
                continue
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        
        # คำนวณค่าเฉลี่ยของเวกเตอร์
        return np.mean(word_vectors, axis=0)

    def load_models(self, models_dir='models'):
        """
        โหลดโมเดลทั้งหมดจากไดเรกทอรี
        รวมถึง TF-IDF vectorizer ถ้ามี
        """
        # รายการโมเดลที่ต้องการโหลด
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'random_forest_tfidf': 'random_forest_tfidf_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl'
        }
        
        # โหลดแต่ละโมเดล
        for model_name, filename in model_files.items():
            file_path = os.path.join(models_dir, filename)
            try:
                with open(file_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"โหลดโมเดล {model_name} สำเร็จ")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการโหลดโมเดล {model_name}: {e}")
        
        # พยายามโหลด TF-IDF vectorizer
        tfidf_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            try:
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                print("โหลด TF-IDF Vectorizer สำเร็จ")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการโหลด TF-IDF Vectorizer: {e}")

    def analyze_sentiment(self, text):
        """
        วิเคราะห์ความรู้สึกของข้อความโดยใช้ทุกโมเดลที่มี
        """
        predictions = {}
        processed_text = self.preprocess_text(text)

        try:
            # ทำนายด้วย Random Forest + Word2Vec
            if 'random_forest' in self.models:
                features = self.extract_features_word2vec(processed_text)
                features = features.reshape(1, -1)
                predictions['random_forest'] = self.models['random_forest'].predict(features)[0]
            
            # ทำนายด้วย Random Forest + TF-IDF
            if 'random_forest_tfidf' in self.models and self.tfidf_vectorizer:
                features = self.tfidf_vectorizer.transform([processed_text])
                predictions['random_forest_tfidf'] = self.models['random_forest_tfidf'].predict(features)[0]
            
            # ทำนายด้วย Gradient Boosting
            if 'gradient_boosting' in self.models:
                features = self.extract_features_word2vec(processed_text)
                features = features.reshape(1, -1)
                predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(features)[0]

        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
            return None

        return predictions

    def analyze_file(self, file_path, text_column='Text', class_column='Class'):
        """
        วิเคราะห์ข้อความทั้งหมดจากไฟล์ Excel และบันทึกผลลัพธ์
        """
        try:
            # อ่านข้อมูลจากไฟล์ Excel
            df = pd.read_excel(file_path)
            print(f"อ่านข้อมูลสำเร็จ: {len(df)} รายการ")
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            if text_column not in df.columns or class_column not in df.columns:
                print(f"ไฟล์ต้องมีคอลัมน์ '{text_column}' และ '{class_column}'")
                return
            
            # สร้างโมเดล Word2Vec จากข้อมูลทั้งหมด
            print("กำลังสร้างโมเดล Word2Vec...")
            self.create_word2vec_model(df[text_column].values)
            
            # สร้าง TF-IDF Vectorizer ถ้าจำเป็น
            if self.tfidf_vectorizer is None:
                print("กำลังสร้าง TF-IDF Vectorizer...")
                self.create_tfidf_vectorizer(df[text_column].values)
            
            # วิเคราะห์ข้อความทั้งหมด
            results = []
            for idx, row in df.iterrows():
                predictions = self.analyze_sentiment(row[text_column])
                if predictions:
                    result = {
                        text_column: row[text_column],
                        'Actual_Class': row[class_column],
                        **{f'Predicted_{k}': v for k, v in predictions.items()}
                    }
                    results.append(result)
                
                if (idx + 1) % 10 == 0:
                    print(f"วิเคราะห์ข้อมูลแล้ว {idx + 1} รายการ")
            
            # สร้าง DataFrame และคำนวณความแม่นยำ
            if results:
                results_df = pd.DataFrame(results)
                
                # คำนวณความแม่นยำของแต่ละโมเดล
                for model_name in ['random_forest', 'random_forest_tfidf', 'gradient_boosting']:
                    pred_col = f'Predicted_{model_name}'
                    if pred_col in results_df.columns:
                        accuracy = (results_df['Actual_Class'] == results_df[pred_col]).mean()
                        print(f"\nความแม่นยำของโมเดล {model_name}: {accuracy:.2%}")
                
                # บันทึกผลการวิเคราะห์
                output_file = 'sentiment_analysis_results.xlsx'
                results_df.to_excel(output_file, index=False)
                print(f"\nบันทึกผลการวิเคราะห์ลงไฟล์ '{output_file}' สำเร็จ")
            else:
                print("\nไม่มีผลการวิเคราะห์ที่จะบันทึก")
                
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการวิเคราะห์ไฟล์: {e}")

def main():
    """
    ฟังก์ชันหลักสำหรับรันโปรแกรม
    """
    # สร้างอินสแตนซ์ของ ThaiSentimentAnalyzer
    # กำหนดขนาด vector และจำนวน features ให้ตรงกับโมเดลที่มีอยู่
    analyzer = ThaiSentimentAnalyzer(vector_size=100, max_features=854)
    
    # โหลดโมเดลทั้งหมด
    analyzer.load_models()
    
    # วิเคราะห์ข้อความจากไฟล์
    analyzer.analyze_file('data/Thai_Sentiment.xlsx')

if __name__ == "__main__":
    main()