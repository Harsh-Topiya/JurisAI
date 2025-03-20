import pandas as pd
import numpy as np
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, accuracy_score
import joblib
import random
import os
from datetime import datetime, timedelta

print("Starting multi-label model training...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
try:
    df = pd.read_csv('data/FIR_Data.csv')
    print("Original dataset loaded successfully!")
except FileNotFoundError:
    print("Error: FIR_Data.csv not found. Make sure the file is in the correct directory.")
    exit()

# Load vectorizer from main model
try:
    model_data = joblib.load('models/ipc_section_detector_model.pkl')
    tfidf_vectorizer = model_data['vectorizer']
    ipc_reference = model_data['ipc_reference']
    print("Vectorizer and reference data loaded from main model")
except FileNotFoundError:
    print("Error: Main model not found. Run model_training.py first.")
    exit()

# Define text preprocessing function
def preprocess_text(text):
    """Preprocess text for machine learning"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (keeping legal terms)
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'against', 'under', 'with'}
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to generate random dates within the last 2 years
def random_date():
    end = datetime.now()
    start = end - timedelta(days=365*2)
    random_days = random.randint(0, (end - start).days)
    return (start + timedelta(days=random_days)).strftime("%d-%m-%Y")

# Function to generate synthetic FIR complaints
def generate_fir_complaint(row):
    """Generate a synthetic FIR complaint based on offense description"""
    offense = row['Offense'] if not pd.isna(row['Offense']) else ""
    description = row['Description'] if not pd.isna(row['Description']) else ""
    
    # Templates for FIR complaints
    templates = [
        "I would like to report that {offense}. {additional_details}",
        "I am filing this complaint regarding an incident where {offense}. {additional_details}",
        "On {date}, I witnessed/experienced {offense}. {additional_details}",
        "This is to report that on {date}, {offense} occurred. {additional_details}",
        "I wish to file an FIR as {offense} has taken place. {additional_details}"
    ]
    
    # Generate additional details
    locations = ['home', 'workplace', 'public place', 'market', 'road', 'bus', 'train', 'restaurant', 'shop']
    times = ['morning', 'afternoon', 'evening', 'night', 'early morning']
    accused_types = ['known to me', 'unknown', 'my neighbor', 'a relative', 'a colleague', 'a stranger']
    evidence_types = ['witnesses', 'documents', 'photographs', 'video recording', 'medical reports', 'physical evidence']
    
    additional_details = " ".join([
        f"The incident happened at {random.choice(locations)}.",
        f"This occurred around {random.choice(times)}.",
        f"The accused is {random.choice(accused_types)}.",
        f"I have evidence in the form of {random.choice(evidence_types)}."
    ])
    
    # Generate a random date
    date = random_date()
    
    # Select a template and fill it
    template = random.choice(templates)
    fir_text = template.format(
        offense=offense.lower() if offense else "the incident described",
        additional_details=additional_details,
        date=date
    )
    
    return fir_text

# Function to assign multiple sections to a complaint
def assign_multiple_sections(complaint, num_sections=2):
    # Randomly select num_sections from the dataset
    section_numbers = [key for key in ipc_reference.keys()]
    if len(section_numbers) < num_sections:
        num_sections = len(section_numbers)
    selected_sections = random.sample(section_numbers, num_sections)
    return selected_sections

# Create a multi-label dataset
print("Creating multi-label dataset...")
multi_label_rows = []
for i in range(1000):  # Create 1000 examples
    # Generate a new FIR complaint
    random_row = df.sample(1).iloc[0]
    complaint = generate_fir_complaint(random_row)
    
    # Assign multiple sections
    sections = assign_multiple_sections(complaint)
    
    multi_label_rows.append({
        'FIR_Complaint': complaint,
        'Sections': sections
    })

multi_label_df = pd.DataFrame(multi_label_rows)
print(f"Created {len(multi_label_df)} multi-label examples")

# Save the multi-label dataset
multi_label_df.to_csv('data/multi_label_dataset.csv', index=False)

# Prepare data for multi-label classification
X_multi = multi_label_df['FIR_Complaint'].apply(preprocess_text)

# Transform the labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform([set(sections) for sections in multi_label_df['Sections']])
print(f"Number of unique IPC sections: {len(mlb.classes_)}")

# Split the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)
print(f"Training set size: {len(X_train_multi)}, Test set size: {len(X_test_multi)}")

# Convert text to TF-IDF features
X_train_multi_tfidf = tfidf_vectorizer.transform(X_train_multi)
X_test_multi_tfidf = tfidf_vectorizer.transform(X_test_multi)

# Train a multi-label model
print("Training multi-label model...")
multi_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
multi_model.fit(X_train_multi_tfidf, y_train_multi)

# Evaluate the model
y_pred_multi = multi_model.predict(X_test_multi_tfidf)
hamming = hamming_loss(y_test_multi, y_pred_multi)
subset_acc = accuracy_score(y_test_multi, y_pred_multi)

print(f"Hamming Loss: {hamming:.4f}")
print(f"Subset Accuracy: {subset_acc:.4f}")

# Save the multi-label model
joblib.dump({
    'model': multi_model,
    'vectorizer': tfidf_vectorizer,
    'mlb': mlb,
    'ipc_reference': ipc_reference
}, 'models/ipc_multi_label_model.pkl')

print("Multi-label model saved successfully to 'models/ipc_multi_label_model.pkl'")
print("Multi-label model training completed successfully!")