import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("Starting enhanced data preparation...")

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
    if pd.isna(url):
        return None
    match = re.search(r'section-(\d+[A-Za-z]*)', str(url))
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
            'offense': row['Offense'] if not pd.isna(row['Offense']) else "Not specified",
            'punishment': row['Punishment'] if not pd.isna(row['Punishment']) else "Not specified",
            'cognizable': row['Cognizable'] if not pd.isna(row['Cognizable']) else "Not specified",
            'bailable': row['Bailable'] if not pd.isna(row['Bailable']) else "Not specified",
            'court': row['Court'] if not pd.isna(row['Court']) else "Not specified"
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

# Enhanced data generation for more realistic FIR complaints

# Function to generate random dates within the last 2 years
def random_date():
    end = datetime.now()
    start = end - timedelta(days=365*2)
    random_days = random.randint(0, (end - start).days)
    date = start + timedelta(days=random_days)
    return date.strftime("%d-%m-%Y")

# Indian names for more realistic complaints
indian_first_names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayush", "Dhruv", "Kabir", "Krishna", 
    "Aanya", "Aadhya", "Ananya", "Pari", "Anika", "Navya", "Diya", "Riya", "Sara", "Kiara",
    "Raj", "Vikram", "Amit", "Rahul", "Sanjay", "Rohit", "Nikhil", "Priya", "Neha", "Pooja", 
    "Sneha", "Meera", "Geeta", "Anjali", "Kiran", "Kavita", "Sunita", "Anita", "Suman", "Radha"
]

indian_last_names = [
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Jain", "Shah", "Mishra", "Chauhan",
    "Yadav", "Patil", "Reddy", "Nair", "Pillai", "Desai", "Mehta", "Joshi", "Malhotra", "Bose",
    "Chatterjee", "Banerjee", "Das", "Dutta", "Mukherjee", "Roy", "Sen", "Kapoor", "Khanna", "Chopra"
]

# Indian locations for more realistic settings
indian_cities = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana"
]

indian_neighborhoods = [
    "Andheri", "Bandra", "Powai", "Juhu", "Dadar", "Malad", "Borivali", "Vasant Kunj", "Hauz Khas", "Dwarka",
    "Indiranagar", "Koramangala", "Jayanagar", "Banjara Hills", "Jubilee Hills", "Adyar", "T. Nagar", "Salt Lake",
    "Alipore", "Kothrud", "Aundh", "Navrangpura", "C.G. Road", "Malviya Nagar", "Vaishali", "Model Town"
]

indian_landmarks = [
    "City Mall", "Central Park", "Metro Station", "Bus Terminal", "Railway Station", "Municipal Corporation",
    "District Court", "General Hospital", "University Campus", "IT Park", "Market Complex", "Stadium",
    "Exhibition Ground", "Cinema Hall", "Community Center", "Police Station", "Post Office", "Bank"
]

# Enhanced list of locations for incidents
locations = {
    'residential': [
        "my home", "my apartment", "my residence", "my flat", "my house", 
        "the victim's house", "the complainant's residence", "the residential complex",
        "my rented accommodation", "my housing society", "my building"
    ],
    'public': [
        "the market", "the shopping mall", "the bus stop", "the railway station", "the metro station",
        "the public park", "the main road", "the highway", "the cinema hall", "the restaurant",
        "the public gathering", "the community center", "the stadium", "the exhibition ground"
    ],
    'workplace': [
        "my office", "my workplace", "the company premises", "the factory", "the warehouse",
        "the construction site", "the business establishment", "the shop", "the commercial complex"
    ],
    'educational': [
        "the school", "the college campus", "the university grounds", "the coaching center",
        "the educational institution", "the library", "the laboratory", "the classroom"
    ],
    'transport': [
        "the bus", "the train", "the auto-rickshaw", "the taxi", "the metro", "the flight",
        "the shared cab", "the public transport", "the ferry", "the private vehicle"
    ]
}

# Time expressions for more natural language
time_expressions = [
    "around {hour}:{minute} {period}", 
    "at approximately {hour}:{minute} {period}",
    "between {hour}:{minute} and {hour2}:{minute2} {period}",
    "shortly after {hour}:{minute} {period}",
    "just before {hour}:{minute} {period}",
    "during the {time_of_day}",
    "in the {time_of_day}",
    "late {time_of_day}",
    "early {time_of_day}"
]

times_of_day = ["morning", "afternoon", "evening", "night"]

# Enhanced accused descriptions
accused_descriptions = {
    'known': [
        "my neighbor {name}", "my colleague {name}", "my acquaintance {name}", 
        "a person known to me, {name}", "my {relation} named {name}", 
        "my former {relation} {name}", "a local resident named {name}",
        "my business associate {name}", "my tenant {name}", "my landlord {name}"
    ],
    'unknown': [
        "an unknown person", "an unidentified individual", "a stranger", 
        "an unknown assailant", "an unidentified man", "an unidentified woman",
        "a group of unknown individuals", "a masked person", "an unknown youth",
        "an unidentified gang", "some unknown miscreants"
    ],
    'descriptions': [
        "wearing a {color} shirt and {color2} pants", 
        "of approximately {height} height and {build} build",
        "aged around {age} years", 
        "with {facial_feature}",
        "driving a {vehicle_color} {vehicle_type}",
        "carrying a {weapon_type}",
        "speaking in {language}"
    ]
}

relations = ["friend", "relative", "brother", "sister", "cousin", "uncle", "aunt", "in-law", "colleague", "classmate"]
colors = ["red", "blue", "black", "white", "green", "yellow", "brown", "grey", "purple", "orange"]
heights = ["tall", "medium", "short"]
builds = ["heavy", "medium", "slim", "muscular", "thin"]
ages = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50", "50-55"]
facial_features = ["a beard", "a mustache", "long hair", "short hair", "a scar on the face", "a tattoo on the arm"]
vehicle_types = ["motorcycle", "car", "scooter", "SUV", "auto-rickshaw", "bicycle", "van", "truck"]
weapon_types = ["knife", "gun", "rod", "stick", "chain", "sharp object", "blunt object"]
languages = ["Hindi", "English", "local dialect", "regional language"]

# Enhanced evidence types
evidence_types = [
    "eyewitness testimony", "CCTV footage", "photographs", "video recordings", 
    "medical reports", "forensic evidence", "documentary proof", "physical evidence",
    "audio recordings", "digital communications", "transaction records", "call records",
    "material evidence", "expert testimony", "recovered stolen items"
]

# Injury descriptions for assault cases
injury_descriptions = [
    "causing injuries to my {body_part}", 
    "resulting in a {injury_type} to my {body_part}",
    "inflicting multiple {injury_type}s on my body",
    "causing me to suffer from {injury_type} and {injury_type2}",
    "leading to severe {injury_type} requiring medical attention",
    "resulting in bleeding from my {body_part}",
    "causing me to lose consciousness due to the {injury_type}"
]

body_parts = ["head", "face", "arm", "leg", "back", "chest", "hand", "eye", "shoulder", "neck", "stomach"]
injury_types = ["bruise", "cut", "fracture", "wound", "laceration", "abrasion", "contusion", "swelling"]

# Theft details for property crimes
stolen_items = [
    "cash worth Rs. {amount}", 
    "gold jewelry worth approximately Rs. {amount}",
    "electronic items including {electronics}",
    "important documents including {documents}",
    "my {item} valued at Rs. {amount}",
    "personal belongings worth Rs. {amount}",
    "household items worth approximately Rs. {amount}"
]

electronics = ["mobile phone", "laptop", "television", "camera", "tablet", "smartwatch", "headphones"]
documents = ["Aadhaar card", "PAN card", "passport", "driving license", "property papers", "certificates"]
items = ["wallet", "purse", "bag", "bicycle", "motorcycle", "watch", "ring", "necklace"]

# Function to generate a random name
def generate_name():
    return f"{random.choice(indian_first_names)} {random.choice(indian_last_names)}"

# Function to generate a random time
def generate_time():
    hour = random.randint(1, 12)
    minute = random.choice(["00", "15", "30", "45"])
    period = random.choice(["AM", "PM"])
    hour2 = hour + random.randint(1, 3)
    if hour2 > 12:
        hour2 = hour2 - 12
        if period == "AM":
            period2 = "PM"
        else:
            period2 = period
    else:
        period2 = period
    
    time_expression = random.choice(time_expressions)
    if "between" in time_expression:
        return time_expression.format(hour=hour, minute=minute, period=period, hour2=hour2, minute2=minute, period2=period2)
    elif "time_of_day" in time_expression:
        return time_expression.format(time_of_day=random.choice(times_of_day))
    else:
        return time_expression.format(hour=hour, minute=minute, period=period)

# Function to generate a random location
def generate_location():
    location_type = random.choice(list(locations.keys()))
    location = random.choice(locations[location_type])
    
    # Sometimes add a city or neighborhood for more detail
    if random.random() < 0.3:
        city = random.choice(indian_cities)
        return f"{location} in {city}"
    elif random.random() < 0.3:
        neighborhood = random.choice(indian_neighborhoods)
        return f"{location} in {neighborhood}"
    elif random.random() < 0.3:
        landmark = random.choice(indian_landmarks)
        return f"{location} near {landmark}"
    else:
        return location

# Function to generate accused description
def generate_accused():
    if random.random() < 0.6:  # 60% chance of known accused
        accused_type = "known"
        name = generate_name()
    else:
        accused_type = "unknown"
        name = ""
    
    accused = random.choice(accused_descriptions[accused_type])
    accused = accused.format(name=name, relation=random.choice(relations))
    
    # Add additional description sometimes
    if random.random() < 0.7:
        description = random.choice(accused_descriptions['descriptions'])
        description = description.format(
            color=random.choice(colors),
            color2=random.choice(colors),
            height=random.choice(heights),
            build=random.choice(builds),
            age=random.choice(ages),
            facial_feature=random.choice(facial_features),
            vehicle_color=random.choice(colors),
            vehicle_type=random.choice(vehicle_types),
            weapon_type=random.choice(weapon_types),
            language=random.choice(languages)
        )
        accused += f", {description}"
    
    return accused

# Function to generate evidence description
def generate_evidence():
    num_evidences = random.randint(1, 3)
    selected_evidences = random.sample(evidence_types, num_evidences)
    
    if len(selected_evidences) == 1:
        return f"I have {selected_evidences[0]} as evidence"
    else:
        return f"I have {', '.join(selected_evidences[:-1])} and {selected_evidences[-1]} as evidence"

# Function to generate injury description for assault cases
def generate_injury():
    description = random.choice(injury_descriptions)
    return description.format(
        body_part=random.choice(body_parts),
        injury_type=random.choice(injury_types),
        injury_type2=random.choice(injury_types)
    )

# Function to generate stolen items description for theft cases
def generate_stolen_items():
    num_items = random.randint(1, 3)
    items_list = []
    
    for _ in range(num_items):
        item_template = random.choice(stolen_items)
        item = item_template.format(
            amount=f"{random.randint(1, 100)},{random.randint(100, 999)}",
            electronics=random.choice(electronics),
            documents=random.choice(documents),
            item=random.choice(items)
        )
        items_list.append(item)
    
    if len(items_list) == 1:
        return items_list[0]
    else:
        return f"{', '.join(items_list[:-1])} and {items_list[-1]}"

# Enhanced function to generate synthetic FIR complaints based on offense type
def generate_enhanced_fir_complaint(row):
    """Generate a realistic synthetic FIR complaint based on offense description and type"""
    offense = row['Offense'] if not pd.isna(row['Offense']) else ""
    description = row['Cleaned_Description'] if not pd.isna(row['Cleaned_Description']) else ""
    section = row['Section_Number'] if not pd.isna(row['Section_Number']) else ""
    
    # Determine complaint type based on offense or section
    offense_lower = offense.lower() if not pd.isna(offense) else ""
    
    # Check for specific offense types to generate tailored complaints
    is_theft = any(word in offense_lower for word in ["theft", "steal", "burglary", "robbery"])
    is_assault = any(word in offense_lower for word in ["assault", "hurt", "injur", "attack", "beat", "wound"])
    is_sexual = any(word in offense_lower for word in ["rape", "sexual", "molest", "outrage", "modesty"])
    is_kidnap = any(word in offense_lower for word in ["kidnap", "abduct", "hostage"])
    is_murder = any(word in offense_lower for word in ["murder", "kill", "death", "homicide"])
    is_fraud = any(word in offense_lower for word in ["fraud", "cheat", "deceive", "misrepresent"])
    is_threat = any(word in offense_lower for word in ["threat", "intimidat", "extort", "blackmail"])
    
    # Complainant details
    complainant_name = generate_name()
    
    # Date and time details
    date = random_date()
    time = generate_time()
    
    # Location details
    location = generate_location()
    
    # Accused details
    accused = generate_accused()
    
    # Evidence details
    evidence = generate_evidence()
    
    # Basic complaint structure
    complaint_parts = [
        f"To the esteemed officer in charge of the police station,",
        f"Subject: Filing of FIR regarding {offense.lower() if offense else 'the incident'}.",
        f"Respected Sir/Madam,",
        f"I, {complainant_name}, wish to lodge a formal complaint regarding {offense.lower() if offense else 'an incident'} that occurred on {date}, {time}, at {location}.",
        f"{accused} "
    ]
    
    # Add specific details based on offense type
    if is_theft:
        stolen_items_desc = generate_stolen_items()
        action_verbs = ["stole", "took away", "robbed me of", "snatched", "made away with"]
        if "robbery" in offense_lower:
            # Add weapon and threat elements for robbery
            weapons = ["knife", "gun", "sharp object", "blunt object"]
            threat_actions = ["threatened me with", "pointed", "brandished", "menacingly displayed"]
            threat_part = f"{random.choice(threat_actions)} a {random.choice(weapons)} and "
            complaint_parts.append(f"{threat_part}{random.choice(action_verbs)} {stolen_items_desc}.")
            if random.random() < 0.7:  # 70% chance to include injury for robbery
                complaint_parts.append(f"During the incident, the accused {generate_injury()}.")
        else:
            complaint_parts.append(f"{random.choice(action_verbs)} {stolen_items_desc}.")
            
    elif is_assault:
        assault_verbs = ["attacked", "assaulted", "hit", "beat", "struck"]
        weapons = ["bare hands", "fists", "a stick", "a rod", "a knife", "a blunt object"]
        complaint_parts.append(f"{random.choice(assault_verbs)} me with {random.choice(weapons)}, {generate_injury()}.")
        
    elif is_sexual:
        # Handle sensitively with appropriate language
        complaint_parts.append("committed a sexual offense against me.")
        complaint_parts.append("I was subjected to this traumatic experience against my will and consent.")
        
    elif is_kidnap:
        kidnap_verbs = ["forcibly took", "abducted", "kidnapped", "forcibly detained"]
        if random.random() < 0.5:  # 50% chance of ransom scenario
            amount = f"{random.randint(1, 50)},{random.randint(10, 99)},{random.randint(100, 999)}"
            complaint_parts.append(f"{random.choice(kidnap_verbs)} me and demanded a ransom of Rs. {amount}.")
        else:
            complaint_parts.append(f"{random.choice(kidnap_verbs)} me and held me against my will.")
            
    elif is_murder:
        murder_verbs = ["killed", "murdered", "caused the death of"]
        victim = f"my {random.choice(['family member', 'relative', 'friend', 'neighbor', 'colleague'])}"
        weapons = ["knife", "gun", "blunt object", "poison", "strangulation"]
        complaint_parts.append(f"{random.choice(murder_verbs)} {victim} using {random.choice(weapons)}.")
        
    elif is_fraud:
        fraud_verbs = ["cheated", "defrauded", "deceived", "misled"]
        fraud_methods = [
            "by falsely promising high returns on investment",
            "through a fake business scheme",
            "by misrepresenting facts about a product",
            "through identity theft",
            "by creating fake documents"
        ]
        amount = f"{random.randint(1, 50)},{random.randint(10, 99)},{random.randint(100, 999)}"
        complaint_parts.append(f"{random.choice(fraud_verbs)} me {random.choice(fraud_methods)}, causing a loss of Rs. {amount}.")
        
    elif is_threat:
        threat_verbs = ["threatened", "intimidated", "blackmailed", "coerced"]
        threat_methods = [
            "with physical harm to me and my family",
            "by saying they would damage my property",
            "by threatening to release private information",
            "with dire consequences",
            "by sending threatening messages"
        ]
        complaint_parts.append(f"{random.choice(threat_verbs)} me {random.choice(threat_methods)}.")
        
    else:
        # Generic offense description for other types
        generic_verbs = ["committed", "perpetrated", "carried out", "engaged in"]
        complaint_parts.append(f"{random.choice(generic_verbs)} {offense.lower() if offense else 'the offense'}.")
    
    # Add evidence statement
    complaint_parts.append(f"{evidence}.")
    
    # Add request for action
    action_requests = [
        "I request that appropriate legal action be taken against the accused.",
        "I urge the authorities to register an FIR and take necessary action.",
        "I request a thorough investigation into this matter and appropriate legal proceedings.",
        "I seek justice and request that the accused be brought to book under relevant sections of law."
    ]
    complaint_parts.append(random.choice(action_requests))
    
    # Combine all parts into a coherent complaint
    fir_complaint = " ".join(complaint_parts)
    
    return fir_complaint

# Continue edge case templates for important IPC sections
# Initialize edge_case_templates as an empty dictionary
edge_case_templates = {}

edge_case_templates.update({
    # Theft and robbery edge cases
    'theft_edge_cases': [
        {
            'template': "I, {name}, wish to report a theft that occurred at my residence in {location}. On {date}, {time}, I returned home to find that my door lock had been broken. Upon inspection, I discovered that {stolen_items}. The intruder(s) had thoroughly ransacked my house, opening drawers and cupboards. I immediately contacted my neighbor {neighbor_name} who confirmed seeing a suspicious person {suspect_description} lurking around our building earlier. I have filed a police complaint at {police_station} and request a thorough investigation. {evidence}.",
            'applicable_sections': ['Section 379', 'Section 380', 'Section 454', 'Section 457']
        },
        {
            'template': "I, {name}, am writing to report a case of pickpocketing. On {date}, {time}, while traveling in a crowded {transport} at {location}, I felt someone bump into me. Shortly after, I realized that {stolen_items}. The suspect was {suspect_description}. I tried to chase the person but lost them in the crowd. Several bystanders witnessed the incident. {evidence}. I request immediate registration of an FIR and appropriate action.",
            'applicable_sections': ['Section 379', 'Section 356']
        }
    ],
    # Robbery edge cases
    'robbery_edge_cases': [
        {
            'template': "I, {name}, wish to report a robbery that occurred at {location}. On {date}, {time}, I was approached by {suspect_description} who threatened me with a {weapon_type}. They demanded that I hand over {stolen_items}. In fear for my life, I complied. The suspect fled towards {landmark}. {evidence}. I request the registration of an FIR and immediate action to apprehend the suspect.",
            'applicable_sections': ['Section 392', 'Section 397', 'Section 398']
        }
    ],
    # Additional edge cases for other types of offenses can be defined similarly
})

# Function to generate FIR based on edge cases
def generate_edge_case_fir_complaint(section, row):
    for category, templates in edge_case_templates.items():
        for template_info in templates:
            if section in template_info['applicable_sections']:
                template = template_info['template']
                # Populate the template with random data where necessary
                fir_complaint = template.format(
                    name=generate_name(),
                    location=generate_location(),
                    date=random_date(),
                    time=generate_time(),
                    stolen_items=generate_stolen_items(),
                    neighbor_name=generate_name(),
                    suspect_description=generate_accused(),
                    police_station=random.choice(indian_landmarks),
                    evidence=generate_evidence(),
                    transport=random.choice(locations['transport']),
                    weapon_type=random.choice(weapon_types),
                    landmark=random.choice(indian_landmarks)
                )
                return fir_complaint
    return None

# Apply the FIR generation to each row in the dataset
df['FIR_Complaint'] = df.apply(lambda row: generate_edge_case_fir_complaint(row['Section_Number'], row) or generate_enhanced_fir_complaint(row), axis=1)

# Duplicate the dataset to generate more permutations
num_permutations = 5  # Number of times to duplicate and modify the dataset
dfs = []

for _ in range(num_permutations):
    df_copy = df.copy()
    
    # Randomize certain columns to create unique permutations
    df_copy['Section_Number'] = df_copy['Section_Number'].apply(lambda x: f"{x}-{random.randint(1, 99)}")
    df_copy['Offense'] = df_copy['Offense'].apply(lambda x: f"{x} (variation {random.randint(1, 10)})" if x else x)
    df_copy['Description'] = df_copy['Description'].apply(lambda x: f"{x} (case {random.randint(1, 100)})" if x else x)
    
    dfs.append(df_copy)

# Combine all permutations into a single DataFrame
df = pd.concat(dfs, ignore_index=True)

# Select only the required columns for the final dataset
columns_to_keep = ['Section_Number', 'FIR_Complaint', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court']
df_final = df[columns_to_keep]

# Save the processed DataFrame with complaints to a new CSV
df_final.to_csv('data/FIRs.csv', index=False)
print("Enhanced FIR data saved to 'data/FIRs.csv' in the specified format.")