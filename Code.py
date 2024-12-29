# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Load the dataset
dataset = pd.read_csv('D:/research/dataset/newdataset.csv')
print(dataset.head())

# Step 2: Data Preprocessing (Text Cleaning)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Clean and preprocess the text
def clean_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

dataset['Cleaned_Resume'] = dataset['Resume'].apply(clean_text)

# Step 3: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['Cleaned_Resume'])

# Step 4: Label Encoding (Converting categories to numeric)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(dataset['Category'])

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training using SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Step 7: Model Evaluation

# Get predictions
y_pred = model.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Fairness Metrics
# Demographic Parity (for each category)
category_counts = dataset['Category'].value_counts(normalize=True)
print(category_counts)

# Save the Model
joblib.dump(model, 'D:/research/Model/fairness_aware_resume_model.joblib')

# Save the TF-IDF Vectorizer
joblib.dump(vectorizer, 'D:/research/Model/tfidf_vectorizer.joblib')

# Save the Label Encoder
joblib.dump(encoder, 'D:/research/Model/label_encoder.joblib')

# Save Fairness Metrics (Demographic Parity)
category_counts.to_csv('D:/research/Model/category_fairness.csv', header=True)
