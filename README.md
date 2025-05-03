# FIR Prediction System 🚔

The **FIR Prediction System** is a web-based application designed to assist law enforcement, legal professionals, and users in analyzing First Information Reports (FIRs). It predicts relevant Indian Penal Code (IPC) sections based on FIR content and provides structured outputs, storing user-specific FIR data in Excel files.

---

## 🌟 Features

- **🔐 Login and Signup**
  - Log in using:
    - Username + password
    - Aadhaar number (12-digit)
  - Strong password validation (uppercase, lowercase, digit, special char)
  - New user registration

- **📄 FIR Analysis**
  - Submit FIR complaint in plain text or voice input
  - Predict relevant IPC sections using machine learning + Gemini API
  - View detailed info:
    - Description, offense, punishment
    - Cognizable/bailable status, court jurisdiction
    - Confidence score

- **📊 Excel File Storage**
  - Each user has their own Excel (e.g., `username_fir_data.xlsx`)
  - Stores FIR complaints + predicted IPC sections


---

## ⚙️ Technologies Used

### Backend
- **Flask** — web framework
- **SQLite** — lightweight user database
- **Python Libraries**:
  - `nltk` → text preprocessing
  - `joblib` → load ML models
  - `openpyxl` → handle Excel files
  - `requests` → call Gemini API

### Frontend
- **HTML/CSS** → responsive, modern glass-effect design
- **JavaScript** → dynamic UI

### Machine Learning
- Pre-trained models (single-label + multi-label) using `scikit-learn`
- Text vectorization using **TF-IDF**

### API Integration
- **Gemini API** → advanced FIR analysis, returns structured JSON

---

## 🔄 Project Workflow

1. **User Authentication**
   - Login with username/password or Aadhaar
   - Password strength validated

2. **FIR Submission**
   - FIR text or voice input submitted
   - Preprocessed (tokenization, stopwords, lemmatization)

3. **Crime Detection**
   - Matches keywords to crime categories (e.g., robbery, assault)

4. **Model Predictions**
   - TF-IDF vectorization → ML model predicts IPC sections
   - Confidence scores generated

5. **Gemini API**
   - FIR sent for advanced analysis, returns IPC sections


6. **Results Display**
   - Shows predicted IPCs + details, generates visual charts

7. **Excel Storage**
   - Stores results in user-specific Excel files

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd fir-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up the database
python
>>> import sqlite3
>>> conn = sqlite3.connect('users.db')
>>> cursor = conn.cursor()
>>> cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    aadhaar TEXT NOT NULL
)
''')
>>> conn.commit()
>>> conn.close()

# Set your Gemini API key (in environment or config)

# Run the app
flask run
```
--- 
## 🌐 Endpoints

### Main Endpoints
| Endpoint     | Description                                    |
|--------------|------------------------------------------------|
| `/login`     | User login                                     |
| `/signup`    | User signup                                    |
| `/predict`   | Predict IPC sections for an FIR                |
| `/gemini`    | Advanced FIR analysis using the Gemini API     |
| `/logout`    | Log out the current user                       |


---

## 🌱 Future Enhancements

- ✅ **Role-Based Access**  
  Add admin and user roles for better access control.

- ✅ **Enhanced Visualizations**  
  Add more charts and graphs for IPC section analysis.

- ✅ **Multi-Language Support**  
  Support FIRs in regional languages using NLP techniques.

- ✅ **OCR Support**  
  Allo users to upload image or pdf of the FIR Complaints.

- ✅ **Mobile App Integration**  
  Develop a mobile app for easier FIR submission and analysis.

- ✅ **Real-Time Notifications**  
  Notify users of updates via email or SMS.

---

## 👥 Contributors

- **Kabir Mota** → Developer and Maintainer

---

## 📄 Copyright

© 2025 Kabir Mota. All rights reserved.

---

## 📬 Need Help?

Feel free to reach out or open an issue!


