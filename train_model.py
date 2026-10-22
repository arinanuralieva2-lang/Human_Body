import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("Загрузка расширенной базы данных (1000+ примеров)...")
df = pd.read_csv('expanded_dataset.csv', engine='python')

# Извлекаем данные
X_texts = df['text'].tolist()
y_specialists = df['specialist'].tolist()
y_urgency = df['urgency'].tolist()

print(f"Всего записей для обучения: {len(X_texts)}")

# 1. Обучаем модель для определения профильного врача
specialist_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Анализирует одиночные слова и пары слов
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
specialist_pipeline.fit(X_texts, y_specialists)

# 2. Обучаем модель для определения степени срочности
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
urgency_pipeline.fit(X_texts, y_urgency)

# 3. Сохраняем обновлённый бандл в hbba_model.pkl
model_bundle = {
    "specialist_model": specialist_pipeline,
    "urgency_model": urgency_pipeline
}

joblib.dump(model_bundle, 'hbba_model.pkl')
print("УСПЕХ! Новая модель обучена на 1200+ записях и сохранена в 'hbba_model.pkl'")