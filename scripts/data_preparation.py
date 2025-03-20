import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from datetime import datetime, timedelta
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("Starting data preparation...")

# Load the dataset
try:
    df = pd.read_csv('data/FIR_Data.csv')
    # Convert NaN values to None (JSON compatible)
    df = df.where(pd.notna(df), None)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: FIR_Data.csv not found. Make sure the file is in the correct directory.")
    exit()

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Function to extract section number from URL
def extract_section_number(url):
    match = re.search(r'section-(\d+)', url)
    if match:
        return f"Section {match.group(1)}"
    return None

# Extract section numbers
df['Section_Number'] = df['URL'].apply(extract_section_number)

# Clean descriptions (remove HTML, URLs, etc.)
def clean_description(text):
    if pd.isna(text):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', str(text))
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned_Description'] = df['Description'].apply(clean_description)

# Create a reference dictionary for IPC sections
ipc_reference = {}
for _, row in df.iterrows():
    if row['Section_Number'] is not None:
        ipc_reference[row['Section_Number']] = {
            'description': row['Cleaned_Description'],
            'offense': row['Offense'],
            'punishment': row['Punishment'],
            'cognizable': row['Cognizable'],
            'bailable': row['Bailable'],
            'court': row['Court']
        }

# Save the reference data
with open('data/ipc_reference.json', 'w') as f:
    json.dump(ipc_reference, f, indent=4)

print("IPC reference data saved to 'data/ipc_reference.json'")

# Text preprocessing function
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

# Apply preprocessing to the dataset
df['Processed_Description'] = df['Cleaned_Description'].apply(preprocess_text)
df['Processed_Offense'] = df['Offense'].apply(preprocess_text)

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
    description = row['Cleaned_Description'] if not pd.isna(row['Cleaned_Description']) else ""
    
    # Templates for FIR complaints
    templates = [
        "I would like to report that {offense}. {additional_details}",
        "I am filing this complaint regarding an incident where {offense}. {additional_details}",
        "On {date}, I witnessed/experienced {offense}. {additional_details}",
        "This is to report that on {date}, {offense} occurred. {additional_details}",
        "I wish to file an FIR as {offense} has taken place. {additional_details}"
    ]
    
    # Extract key phrases from the description
    description_clean = description.replace("Description of IPC Section", "")
    description_clean = description_clean.replace("IPC", "")
    description_clean = description_clean.replace("in Simple Words", "")
    description_clean = re.sub(r'\d+', '', description_clean).strip()
    
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

# Generate synthetic FIR complaints
df['FIR_Complaint'] = df.apply(generate_fir_complaint, axis=1)

# Create multiple variations for each section to improve training
expanded_rows = []
for _, row in df.iterrows():
    section = row['Section_Number']
    if section is not None:
        # Create 5 variations of FIR complaints for each section
        for i in range(5):
            new_row = {
                'Section_Number': section,
                'FIR_Complaint': generate_fir_complaint(row),
                'Offense': row['Offense'],
                'Punishment': row['Punishment'],
                'Cognizable': row['Cognizable'],
                'Bailable': row['Bailable'],
                'Court': row['Court']
            }
            expanded_rows.append(new_row)

# Create expanded dataframe
expanded_df = pd.DataFrame(expanded_rows)

# Save the synthetic dataset
expanded_df.to_csv('data/synthetic_fir_dataset.csv', index=False)

print(f"Created synthetic dataset with {len(expanded_df)} FIR complaints")
print("\nSample synthetic FIR complaint:")
print(expanded_df['FIR_Complaint'].iloc[0])
print("\nCorresponding IPC section:", expanded_df['Section_Number'].iloc[0])

print("Data preparation completed successfully!")
