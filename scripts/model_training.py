import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SMOTE - install if not available
try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    print("imblearn not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE
    has_smote = True

print("Starting balanced model training for all crime types...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the synthetic dataset
try:
    expanded_df = pd.read_csv('data/synthetic_fir_dataset.csv')
    print(f"Synthetic dataset loaded with {len(expanded_df)} records")
except FileNotFoundError:
    print("Error: synthetic_fir_dataset.csv not found. Run data_preparation.py first.")
    exit()

# Load IPC reference data
try:
    with open('data/ipc_reference.json', 'r') as f:
        ipc_reference = json.load(f)
    print("IPC reference data loaded successfully")
except FileNotFoundError:
    print("Error: ipc_reference.json not found. Run data_preparation.py first.")
    exit()

# Define comprehensive crime mappings to IPC sections
common_crime_sections = {
    'robbery': ['Section 390', 'Section 392', 'Section 394', 'Section 397'],
    'theft': ['Section 378', 'Section 379', 'Section 380', 'Section 381'],
    'assault': ['Section 351', 'Section 352', 'Section 323', 'Section 324', 'Section 325'],
    'kidnapping': ['Section 359', 'Section 360', 'Section 363', 'Section 364', 'Section 365'],
    'murder': ['Section 302', 'Section 303', 'Section 304', 'Section 307'],
    'rape': ['Section 375', 'Section 376', 'Section 376A', 'Section 376D'],
    'sexual_assault': ['Section 354', 'Section 354A', 'Section 354B', 'Section 509'],
    'fraud': ['Section 415', 'Section 420', 'Section 421'],
    'criminal_intimidation': ['Section 503', 'Section 506', 'Section 507'],
    'hurt': ['Section 319', 'Section 320', 'Section 323', 'Section 324', 'Section 325'],
    'wrongful_restraint': ['Section 339', 'Section 340', 'Section 341'],
    'wrongful_confinement': ['Section 340', 'Section 342', 'Section 343', 'Section 344'],
    'cheating': ['Section 415', 'Section 420']
}

# Create a mapping of IPC sections to their categories
def categorize_ipc_section(section_num):
    """Categorize IPC sections into main legal categories"""
    try:
        if isinstance(section_num, str) and section_num.startswith("Section "):
            section_num = int(section_num.replace("Section ", ""))
        else:
            return "Other"
        
        if 299 <= section_num <= 377:
            return "Crimes Against Person"
        elif 378 <= section_num <= 462:
            return "Property Crimes"
        elif 141 <= section_num <= 160:
            return "Public Order"
        elif 120 <= section_num <= 124:
            return "Offenses Against State"
        elif 191 <= section_num <= 229:
            return "False Evidence and Offenses Against Public Justice"
        elif 339 <= section_num <= 348:
            return "Wrongful Restraint and Wrongful Confinement"
        elif 359 <= section_num <= 369:
            return "Kidnapping and Abduction"
        elif 375 <= section_num <= 376:
            return "Sexual Offenses"
        elif 354 <= section_num <= 354:
            return "Sexual Harassment"
        elif 390 <= section_num <= 402:
            return "Robbery and Dacoity"
        elif 378 <= section_num <= 382:
            return "Theft"
        elif 503 <= section_num <= 510:
            return "Criminal Intimidation"
        elif 319 <= section_num <= 338:
            return "Hurt and Grievous Hurt"
        else:
            return "Other"
    except:
        return "Other"

# Add category to reference data
for section, details in ipc_reference.items():
    details['category'] = categorize_ipc_section(section)

# Add category to expanded dataframe
expanded_df['Category'] = expanded_df['Section_Number'].apply(categorize_ipc_section)

# Add specific training examples for various crime types to ensure balanced representation
additional_examples = [
    # Robbery examples
    {
        'FIR_Complaint': "I was returning home from the market when two men on a motorcycle stopped me. One of them pulled out a knife and threatened me. They snatched my bag containing money and documents, hit me on the head, and rode away.",
        'Section_Number': "Section 392",  # Robbery
        'Category': "Robbery and Dacoity",
        'Offense': "Robbery",
        'Punishment': "Rigorous imprisonment for 10 years and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Magistrate First Class"
    },
    
    # Rape examples
    {
        'FIR_Complaint': "I was forcibly taken to an isolated location by a man I knew from college. He locked the doors, threatened me with a knife, and raped me against my will. He also threatened to harm my family if I reported the incident.",
        'Section_Number': "Section 376",  # Rape
        'Category': "Sexual Offenses",
        'Offense': "Rape",
        'Punishment': "Rigorous imprisonment for 10 years to life and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Court of Session"
    },
    {
        'FIR_Complaint': "I was returning home when my neighbor forcibly entered my house and sexually assaulted me. He threatened to kill me if I told anyone about the incident.",
        'Section_Number': "Section 376",  # Rape
        'Category': "Sexual Offenses",
        'Offense': "Rape",
        'Punishment': "Rigorous imprisonment for 10 years to life and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Court of Session"
    },
    
    # Sexual harassment examples
    {
        'FIR_Complaint': "My colleague at work has been repeatedly making unwelcome sexual advances despite my clear refusal. He touches me inappropriately and makes lewd comments about my appearance.",
        'Section_Number': "Section 354A",  # Sexual harassment
        'Category': "Sexual Harassment",
        'Offense': "Sexual harassment",
        'Punishment': "Imprisonment up to 3 years or fine or both",
        'Cognizable': "Cognizable",
        'Bailable': "Bailable",
        'Court': "Any Magistrate"
    },
    
    # Kidnapping examples
    {
        'FIR_Complaint': "My 10-year-old son was kidnapped while returning from school. The kidnappers called and demanded a ransom of Rs. 5 lakhs for his release.",
        'Section_Number': "Section 363",  # Kidnapping
        'Category': "Kidnapping and Abduction",
        'Offense': "Kidnapping from lawful guardianship",
        'Punishment': "Imprisonment for 7 years and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Magistrate First Class"
    },
    
    # Murder examples
    {
        'FIR_Complaint': "My brother was attacked by a group of men with knives and iron rods. He was severely injured and died at the hospital. The attack was due to a land dispute.",
        'Section_Number': "Section 302",  # Murder
        'Category': "Crimes Against Person",
        'Offense': "Murder",
        'Punishment': "Death or life imprisonment and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Court of Session"
    },
    
    # Theft examples
    {
        'FIR_Complaint': "Someone broke into my house when we were away and stole jewelry, cash, and electronic items worth approximately Rs. 2 lakhs.",
        'Section_Number': "Section 380",  # Theft in dwelling house
        'Category': "Property Crimes",
        'Offense': "Theft in dwelling house",
        'Punishment': "Imprisonment for 7 years and fine",
        'Cognizable': "Cognizable",
        'Bailable': "Non-Bailable",
        'Court': "Any Magistrate"
    }
]

# Add examples multiple times for balance (3 times each)
for _ in range(3):
    for example in additional_examples:
        expanded_df = expanded_df._append(example, ignore_index=True)

print(f"Added balanced examples for various crime types. New dataset size: {len(expanded_df)}")

# Examine category distribution
category_counts = expanded_df['Category'].value_counts()
print("\nCategory distribution:")
print(category_counts)

# Plot category distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=expanded_df['Category'])
plt.title('Distribution of IPC Categories')
plt.tight_layout()
plt.savefig('plots/category_distribution.png')
plt.close()

# Enhanced text preprocessing function with comprehensive crime term preservation
def preprocess_text(text):
    """Preprocess text for machine learning with comprehensive crime term preservation"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Preserve common legal phrases before removing punctuation
    text = text.replace("section ", "section_")
    
    # Replace specific crime terms with tokens
    crime_terms = {
        # Sexual offenses
        "rape": "rape_offense",
        "raped": "rape_offense",
        "sexual": "sexual_offense",
        "sexually": "sexual_offense",
        "molest": "sexual_offense",
        "molestation": "sexual_offense",
        "assault": "assault_offense",
        "assaulted": "assault_offense",
        "forced": "forced_act",
        "forcibly": "forced_act",
        "consent": "consent_issue",
        "against will": "consent_issue",
        "outrage": "outrage_modesty",
        "modesty": "outrage_modesty",
        
        # Robbery and theft related
        "robbed": "robbery_crime",
        "robbery": "robbery_crime",
        "snatched": "robbery_crime",
        "snatching": "robbery_crime",
        "knife": "weapon_knife",
        "knife-point": "weapon_knife",
        "gun": "weapon_gun",
        "pistol": "weapon_gun",
        "weapon": "weapon_used",
        "motorcycle": "vehicle_motorcycle",
        "bike": "vehicle_motorcycle",
        "money": "property_money",
        "cash": "property_money",
        "wallet": "property_wallet",
        "bag": "property_bag",
        "purse": "property_purse",
        "documents": "property_documents",
        "jewellery": "property_jewellery",
        "gold": "property_gold",
        "phone": "property_phone",
        "mobile": "property_phone",
        
        # Physical violence
        "hit": "physical_assault",
        "pushed": "physical_assault",
        "injured": "physical_injury",
        "injury": "physical_injury",
        "hurt": "physical_injury",
        "beaten": "physical_assault",
        "beat": "physical_assault",
        "attacked": "physical_assault",
        
        # Threats and intimidation
        "threatened": "criminal_intimidation",
        "threat": "criminal_intimidation",
        "intimidated": "criminal_intimidation",
        "fear": "criminal_intimidation",
        "warned": "criminal_intimidation",
        
        # Kidnapping
        "kidnapping": "kidnap_offense",
        "abduction": "kidnap_offense",
        "abducted": "kidnap_offense",
        "kidnapped": "kidnap_offense",
        "ransom": "kidnap_ransom",
        
        # Homicide
        "murder": "murder_offense",
        "killed": "murder_offense",
        "death": "murder_death",
        
        # Fraud and cheating
        "fraud": "fraud_offense",
        "cheated": "fraud_offense",
        "cheating": "fraud_offense",
        "forgery": "forgery_offense",
        "forged": "forgery_offense",
        
        # Other crimes
        "defamation": "defamation_offense",
        "trespass": "trespass_offense",
        "grievous": "grievous_hurt",
        "wrongful": "wrongful_act",
        "confinement": "confinement_offense",
        "restraint": "restraint_offense",
        "detained": "restraint_offense",
        "locked": "confinement_offense"
    }
    
    for term, replacement in crime_terms.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (keeping legal terms)
    legal_stopwords = {'not', 'no', 'against', 'under', 'with', 'without', 'by', 'to', 'from', 
                      'between', 'within', 'upon', 'near', 'at', 'before', 'after'}
    stop_words = set(stopwords.words('english')) - legal_stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Prepare data for training with improved preprocessing
X = expanded_df['FIR_Complaint'].apply(preprocess_text)
y = expanded_df['Section_Number']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Create TF-IDF features with better parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=12000,  # Increased features
    ngram_range=(1, 3),  # Include up to trigrams
    min_df=2,            # Minimum document frequency
    max_df=0.9,          # Maximum document frequency
    sublinear_tf=True    # Apply sublinear tf scaling
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Apply SMOTE for handling class imbalance if available
if has_smote:
    print("Applying SMOTE to handle class imbalance...")
    try:
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
        print(f"After SMOTE - Training set shape: {X_train_smote.shape}")
        
        # Use the SMOTE-resampled data for training
        X_train_tfidf_final = X_train_smote
        y_train_final = y_train_smote
    except Exception as e:
        print(f"Error applying SMOTE: {str(e)}")
        print("Continuing with original data...")
        X_train_tfidf_final = X_train_tfidf
        y_train_final = y_train
else:
    print("SMOTE not available, continuing with original data.")
    X_train_tfidf_final = X_train_tfidf
    y_train_final = y_train

# Train Random Forest with optimized parameters
print("\nTraining optimized Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf_final, y_train_final)
rf_pred = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Define crime-specific related sections with comprehensive coverage
crime_related_sections = {
    # Rape and sexual offenses
    'Section 376': ['Section 354', 'Section 506', 'Section 342', 'Section 323'],  # Rape with related sections
    'Section 354': ['Section 354A', 'Section 354B', 'Section 509', 'Section 506'],  # Sexual assault
    'Section 354A': ['Section 354', 'Section 509', 'Section 506'],  # Sexual harassment
    
    # Robbery and theft
    'Section 392': ['Section 394', 'Section 397', 'Section 323', 'Section 506'],  # Robbery with related sections
    'Section 394': ['Section 392', 'Section 397', 'Section 323', 'Section 506'],  # Robbery with hurt
    'Section 379': ['Section 378', 'Section 380', 'Section 411'],  # Theft with related sections
    
    # Assault and hurt
    'Section 323': ['Section 324', 'Section 341', 'Section 506'],  # Hurt with related sections
    'Section 324': ['Section 323', 'Section 341', 'Section 506'],  # Hurt by dangerous weapon
    
    # Kidnapping
    'Section 363': ['Section 364', 'Section 365', 'Section 342', 'Section 506'],  # Kidnapping with related sections
    'Section 364': ['Section 363', 'Section 365', 'Section 342', 'Section 506'],  # Kidnapping for ransom
    
    # Murder
    'Section 302': ['Section 307', 'Section 201', 'Section 34', 'Section 120B'],  # Murder with related sections
    'Section 307': ['Section 323', 'Section 324', 'Section 506'],  # Attempt to murder
    
    # Criminal intimidation
    'Section 506': ['Section 323', 'Section 341', 'Section 504'],  # Criminal intimidation
    
    # Wrongful confinement
    'Section 342': ['Section 341', 'Section 323', 'Section 506'],  # Wrongful confinement
}

# Function to assign multiple sections based on crime type
def assign_related_sections(section, num_related=3):
    """Assign related IPC sections based on the primary section"""
    try:
        # If we have predefined related sections for this crime
        if section in crime_related_sections:
            related = crime_related_sections[section]
            return [section] + related[:num_related]
        
        # Otherwise use category-based approach
        primary_category = categorize_ipc_section(section)
        
        # Get sections from the same category
        same_category_sections = [s for s in ipc_reference.keys() 
                                if ipc_reference[s].get('category') == primary_category and s != section]
        
        # If not enough sections in same category, get from other categories
        if len(same_category_sections) < num_related:
            other_sections = [s for s in ipc_reference.keys() if s != section]
            same_category_sections.extend(other_sections)
        
        # Select random related sections
        related_sections = random.sample(same_category_sections, min(num_related, len(same_category_sections)))
        
        # Return primary section plus related sections
        return [section] + related_sections
    except:
        # Fallback if there's an error
        return [section]

# Create multi-label dataset with balanced representation
multi_label_data = []

# Add standard examples
for _, row in expanded_df.sample(n=min(2000, len(expanded_df)), random_state=42).iterrows():
    section = row['Section_Number']
    sections = assign_related_sections(section)
    
    multi_label_data.append({
        'FIR_Complaint': row['FIR_Complaint'],
        'Sections': sections
    })

# Add specific examples for rape cases with multiple sections
rape_examples = [
    {
        'FIR_Complaint': "I was forcibly taken to an isolated location by a man I knew from college. He locked the doors, threatened me with a knife, and raped me against my will. He also threatened to harm my family if I reported the incident.",
        'Sections': ['Section 376', 'Section 342', 'Section 506', 'Section 354']  # Rape, wrongful confinement, criminal intimidation, assault
    },
    {
        'FIR_Complaint': "I was returning home when my neighbor forcibly entered my house and sexually assaulted me. He threatened to kill me if I told anyone about the incident.",
        'Sections': ['Section 376', 'Section 452', 'Section 506', 'Section 354']  # Rape, house trespass, criminal intimidation, assault
    },
    {
        'FIR_Complaint': "My colleague offered me a ride home after work. Instead of taking me home, he drove me to an isolated place, assaulted me physically, and raped me. He threatened to ruin my reputation if I reported him.",
        'Sections': ['Section 376', 'Section 365', 'Section 323', 'Section 506']  # Rape, kidnapping, hurt, criminal intimidation
    }
]

# Add robbery examples
robbery_examples = [
    {
        'FIR_Complaint': "I was returning home from the market when two men on a motorcycle stopped me. One of them pulled out a knife and threatened me. They snatched my bag containing money and documents, hit me on the head, and rode away.",
        'Sections': ['Section 392', 'Section 394', 'Section 506']  # Robbery, causing hurt in robbery, criminal intimidation
    },
    {
        'FIR_Complaint': "Two men on a motorcycle robbed me at knife-point. They took my bag containing Rs. 12,000 and my documents. They hit me and pushed me to the ground before escaping.",
        'Sections': ['Section 392', 'Section 394', 'Section 506']
    }
]

# Add examples multiple times for emphasis (5 times each)
for _ in range(5):
    for example in rape_examples + robbery_examples:
        multi_label_data.append(example)

multi_label_df = pd.DataFrame(multi_label_data)

# Transform to multi-label format
X_multi = multi_label_df['FIR_Complaint'].apply(preprocess_text)
sections_list = [set(sections) for sections in multi_label_df['Sections']]

mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(sections_list)
print(f"Multi-label target shape: {y_multi.shape}")

# Split multi-label data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Transform text to TF-IDF
X_train_multi_tfidf = tfidf_vectorizer.transform(X_train_multi)
X_test_multi_tfidf = tfidf_vectorizer.transform(X_test_multi)

# Train multi-label model with improved parameters
print("Training multi-label model...")
multi_model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    n_jobs=-1
)
multi_model.fit(X_train_multi_tfidf, y_train_multi)

# Evaluate multi-label model
y_pred_multi = multi_model.predict(X_test_multi_tfidf)
from sklearn.metrics import hamming_loss, accuracy_score

hamming = hamming_loss(y_test_multi, y_pred_multi)
subset_acc = accuracy_score(y_test_multi, y_pred_multi)

print(f"Multi-label model - Hamming Loss: {hamming:.4f}")
print(f"Multi-label model - Subset Accuracy: {subset_acc:.4f}")

# Create a combined model package with both single-label and multi-label capabilities
model_package = {
    'single_model': rf_model,  # Using Random Forest as the best model
    'multi_model': multi_model,
    'vectorizer': tfidf_vectorizer,
    'mlb': mlb,
    'ipc_reference': ipc_reference,
    'common_crime_sections': common_crime_sections,
    'crime_related_sections': crime_related_sections
}

# Save the balanced model package
joblib.dump(model_package, 'models/balanced_ipc_model.pkl')

print("Balanced model package saved successfully to 'models/balanced_ipc_model.pkl'")

# Test the models with various crime cases
test_cases = [
    # Rape case
    "I, Priya Sharma, a resident of Gomti Nagar, Lucknow, wish to file a complaint regarding a serious crime committed against me on March 17, 2025. At around 7:30 PM, I was returning home from Phoenix Mall, Lucknow, after meeting a friend. I was walking towards the auto-rickshaw stand near Hazratganj Main Road when Ramesh Yadav, whom I knew from college, forcibly pulled me into his white Honda City (UP32 AB 1234). He drove me to an isolated farmhouse in the outskirts of Lucknow (near Sultanpur Road). Inside the house, he locked the doors and physically assaulted me, threatened me with a knife, and raped me against my will. He warned me not to report to the police, saying that he had political connections and would harm my family if I spoke out.",
    
    # Robbery case
    "Sir, my name is Ramprasad Yadav, and I am a farmer from Rajpura village, Uttar Pradesh. I have come here to file a complaint about something terrible that happened to me. On March 17, 2025, around 8:00 PM, I was returning home from the weekly market in town after selling some of my crops. I was walking alone on the Rajpura-Kanpur road, carrying the money I had earned that day. Suddenly, two men on a black motorcycle with no number plate came from behind. One of them was wearing a red shirt and had his face covered with a cloth, and the other had a black jacket and helmet. They stopped near me and asked for directions to a nearby village. As I was speaking, the man in the red shirt suddenly pulled out a knife and threatened me. Before I could react, the other man snatched my cloth bag containing ₹12,000 in cash, my land documents, and my Aadhaar card. When I tried to shout for help, they hit me on the head, pushed me to the ground, and rode away towards the highway. I was left injured and scared on the roadside.",
    
    # Theft case
    "I would like to report a theft that occurred at my residence yesterday. When I returned home from work, I found that the lock on my front door had been broken. Upon entering, I discovered that my laptop, cash worth Rs. 25,000, and some gold jewelry were missing. The house was in disarray, indicating that the thief had searched through my belongings."
]

print("\nTesting models with various crime cases:")
for i, case in enumerate(test_cases):
    print(f"\nTest Case {i+1}: {case[:100]}...")
    
    # Preprocess the text
    processed_case = preprocess_text(case)
    case_features = tfidf_vectorizer.transform([processed_case])
    
    # Single-label prediction
    single_pred = rf_model.predict(case_features)[0]
    print(f"Single-label prediction: {single_pred}")
    
    # Get top 3 predictions with probabilities
    if hasattr(rf_model, 'predict_proba'):
        proba = rf_model.predict_proba(case_features)[0]
        top_indices = proba.argsort()[-5:][::-1]  # Get top 5
        top_classes = rf_model.classes_[top_indices]
        top_probas = proba[top_indices]
        
        print("\nTop 5 single-label predictions:")
        for cls, prob in zip(top_classes, top_probas):
            print(f"{cls}: {prob:.4f}")
    
    # Multi-label prediction
    multi_pred_binary = multi_model.predict(case_features)
    multi_pred_sections = mlb.inverse_transform(multi_pred_binary)[0]
    print(f"\nMulti-label prediction: {', '.join(multi_pred_sections)}")

print("\nBalanced model training completed successfully!")