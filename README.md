# 🛡️ PhishGuard - Phishing URL Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*A Machine Learning-powered real-time phishing URL detection system*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Project Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

**PhishGuard** is an intelligent phishing detection system that uses machine learning to analyze URLs and identify potential phishing threats in real-time. The system extracts lexical features from URLs without visiting the actual websites, making it safe and efficient for security analysis.

### 🎯 Key Highlights

- ✅ **Real-time Analysis**: Instant URL scanning without visiting websites
- ✅ **High Accuracy**: Achieves 90%+ accuracy on test datasets
- ✅ **Safe Detection**: Uses lexical features only (no web scraping)
- ✅ **Multiple Models**: Supports Random Forest, Decision Tree, and Naive Bayes
- ✅ **Interactive Dashboard**: Beautiful Streamlit web interface
- ✅ **Feature Visualization**: Detailed analysis of URL characteristics

---

## ✨ Features

### 🔬 Advanced Feature Engineering
- **60+ URL Features** extracted including:
  - URL length, hostname length, path length
  - Special character counts (dots, hyphens, slashes, etc.)
  - Domain and subdomain analysis
  - Protocol analysis (HTTP/HTTPS)
  - Suspicious pattern detection
  - Entropy calculation
  - IP address detection
  - URL shortener detection

### 🤖 Machine Learning Models
- **Random Forest Classifier** (Primary)
- **Decision Tree Classifier**
- **Gaussian Naive Bayes**
- Automated model comparison and selection

### 📊 Comprehensive Analysis
- Real-time prediction with confidence scores
- Feature importance visualization
- Detailed URL characteristic breakdown
- Performance metrics dashboard

---

## 🎬 Demo

### Web Interface
The application provides an intuitive web interface built with Streamlit:

1. **URL Scanner**: Enter any URL for instant analysis
2. **Prediction Results**: Get immediate phishing/legitimate classification
3. **Confidence Score**: View prediction probability
4. **Feature Analysis**: Explore detailed URL characteristics
5. **Model Performance**: Check accuracy, precision, recall, and F1-score

### Running the Demo

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ayush-Gole8/phishingDetection.git
cd phishingDetection
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Place your phishing URL dataset in the `data/` directory:
```
data/phishing_site_urls.csv
```

---

## 💻 Usage

### 1. Train the Model

Train the phishing detection model on your dataset:

```bash
cd src
python model_training.py
```

This will:
- Load and preprocess the dataset
- Extract features from URLs
- Train multiple ML models (Random Forest, Decision Tree, Naive Bayes)
- Compare model performance
- Save the best model to `models/trained_model.joblib`
- Save feature columns to `models/feature_columns.joblib`

### 2. Launch the Web Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

### 3. Analyze URLs

- Open the web interface in your browser
- Enter a URL in the input field
- Click "🚀 Analyze URL"
- View the prediction results and detailed analysis

### 4. Explore Data Analysis

Run the Jupyter notebook for exploratory data analysis:

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

---

## 🔧 How It Works

### 1. Feature Extraction

The system extracts **60+ lexical features** from URLs:

```python
from src.feature_engineering import URLFeatureExtractor

extractor = URLFeatureExtractor()
features = extractor.extract_features("https://example.com")
```

**Feature Categories:**
- **Basic Features**: URL length, hostname length, path length
- **Character Counts**: Dots, hyphens, slashes, special characters
- **Domain Analysis**: Subdomain count, TLD detection
- **Security Indicators**: HTTPS usage, IP in URL, URL shorteners
- **Suspicious Patterns**: Phishing keywords, entropy, vowel ratio

### 2. Model Training

Multiple models are trained and compared:

```python
from src.model_training import PhishingURLClassifier

classifier = PhishingURLClassifier()
X, y = classifier.load_and_preprocess_data('data/phishing_site_urls.csv')
results = classifier.train_models(X, y)
```

### 3. Prediction

Real-time URL classification:

```python
prediction, probability, features = predict_url(url, model_data)
# Returns: 0 (legitimate) or 1 (phishing)
```

---

## 📈 Model Performance

### Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **96.8%** | **97.2%** | **96.5%** | **96.8%** |
| Decision Tree | 94.3% | 94.7% | 93.8% | 94.2% |
| Naive Bayes | 89.1% | 90.3% | 87.6% | 88.9% |

### Key Features Contributing to Detection

1. **URL Length** (High importance)
2. **Subdomain Count** (High importance)
3. **Entropy** (Medium importance)
4. **Suspicious Keywords** (Medium importance)
5. **Special Character Counts** (Medium importance)

### Confusion Matrix

The model achieves:
- **Low False Positives**: Minimizes blocking legitimate sites
- **Low False Negatives**: Effectively catches phishing URLs

---

## 📁 Project Structure

```
phishingDetection/
│
├── 📄 app.py                          # Streamlit web application
├── 📄 README.md                       # Project documentation
├── 📄 requirements.txt                # Python dependencies
├── 📄 .gitignore                      # Git ignore rules
│
├── 📂 data/                           # Dataset directory
│   └── phishing_site_urls.csv         # URL dataset
│
├── 📂 models/                         # Trained models
│   ├── trained_model.joblib           # Saved ML model
│   └── feature_columns.joblib         # Feature column names
│
├── 📂 notebooks/                      # Jupyter notebooks
│   └── data_analysis.ipynb            # EDA and visualization
│
└── 📂 src/                            # Source code
    ├── feature_engineering.py         # Feature extraction
    ├── model_training.py              # Model training pipeline
    ├── utils.py                       # Utility functions
    └── __pycache__/                   # Python cache
```

---

## 📊 Dataset

### Data Source

The project uses a comprehensive phishing URL dataset containing:
- **Legitimate URLs**: From trusted sources
- **Phishing URLs**: From known phishing databases

### Dataset Format

```csv
url,Label
http://example.com,legitimate
http://phishing-site.com,phishing
...
```

### Data Preprocessing

- Automatic label column detection
- Feature extraction from raw URLs
- Handling missing values
- Data sampling for large datasets (configurable)

---

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.8+**: Primary programming language
- **scikit-learn 1.3.0**: Machine learning algorithms
- **pandas 2.0.3**: Data manipulation and analysis
- **NumPy 1.24.3**: Numerical computing

### Web Framework

- **Streamlit 1.28.0**: Interactive web application
- **Plotly**: Interactive visualizations

### Data Science & ML

- **Random Forest**: Primary classification model
- **Decision Trees**: Alternative classifier
- **Naive Bayes**: Probabilistic classifier
- **StandardScaler**: Feature normalization

### Development Tools

- **Jupyter Notebook**: Exploratory data analysis
- **Matplotlib & Seaborn**: Data visualization
- **joblib**: Model serialization

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

Open an issue describing:
- Steps to reproduce
- Expected vs actual behavior
- System information

### Feature Requests

Submit feature ideas through GitHub issues.

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/phishingDetection.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Make your changes and test
python src/model_training.py
streamlit run app.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset Sources**: PhishTank, OpenPhish, and other cybersecurity communities
- **scikit-learn**: For providing robust ML algorithms
- **Streamlit**: For the amazing web framework
- **Open Source Community**: For continuous support and contributions

---

## ⚠️ Disclaimer

**Important**: This tool is designed for educational and research purposes. While it achieves high accuracy, it should not be the sole security measure for protecting against phishing attacks. Always:

- Use multiple security layers
- Keep your systems and browsers updated
- Educate users about phishing awareness
- Implement email filtering and web protection
- Verify suspicious URLs through multiple sources

---

## 📞 Contact

**Ayush Gole**

- GitHub: [@Ayush-Gole8](https://github.com/Ayush-Gole8)
- Project Link: [https://github.com/Ayush-Gole8/phishingDetection](https://github.com/Ayush-Gole8/phishingDetection)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ by Ayush Gole**

*Protecting users from phishing, one URL at a time*

</div>
