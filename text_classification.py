# text_classification.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 示例數據
data = {
    'text': ['I love this movie', 'This movie is terrible', 'Great film!', 'Worst movie ever', 'Amazing plot', 'Not good'],
    'label': [1, 0, 1, 0, 1, 0]
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 特徵提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))