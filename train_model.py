import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Готовый обучающий датасет (симптомы -> врач и уровень срочности)
data = [
    # Мигрень и головные боли
    ("пульсирующая боль в виске тошнота свет режет глаза", "Невролог", "Плановый визит"),
    ("давит затылок шумит в ушах боль при повороте шеи", "Невролог", "Плановый визит"),
    ("стреляющая боль в голове головокружение слабость", "Невролог", "Плановый визит"),
    
    # Опасные состояния (Красные флаги)
    ("внезапная громоподобная боль онемение руки нарушение речи", "Скорая помощь (103)", "ЭКСТРЕННО"),
    ("резкая острая боль за грудиной онемение левой руки под лопаткой", "Скорая помощь (103)", "ЭКСТРЕННО"),
    
    # Позвоночник и суставы
    ("острая боль в пояснице отдаёт в ногу тяжело разогнуться", "Травматолог / Вертебролог", "Плановый визит"),
    ("боль в коленном суставе хруст при ходьбе отечность", "Травматолог-ортопед", "Плановый визит"),
    ("Боль в грудном отделе при глубоком вдохе межреберная", "Невролог / Терапевт", "Плановый визит")
]

# Разделяем данные
X_texts = [item[0] for item in data]
y_specialists = [item[1] for item in data]
y_urgency = [item[2] for item in data]

# 2. Обучаем модель для определения врача
specialist_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])
specialist_pipeline.fit(X_texts, y_specialists)

# 3. Обучаем модель для определения срочности
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])
urgency_pipeline.fit(X_texts, y_urgency)

# 4. Сохраняем обе модели в один файл hbba_model.pkl
model_bundle = {
    "specialist_model": specialist_pipeline,
    "urgency_model": urgency_pipeline
}

joblib.dump(model_bundle, 'hbba_model.pkl')
print("УСПЕХ! Модель обучена и сохранена в файл 'hbba_model.pkl'")