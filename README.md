# Recommender System App

## Financial Literacy Content-Based Recommendation System

##  Overview
A sophisticated content-based recommendation system for financial literacy content that combines text analysis (TF-IDF) with categorical feature encoding to deliver personalized educational recommendations. The system handles both existing users and cold-start scenarios with an intuitive Streamlit web interface.



## 🚀 Getting Started

### Quick Start (For Users)
**You can run the main application immediately without running any other files:**

```bash
streamlit run code/main.py
```


The system will automatically:

Load your data from data/cleaned_financial_data.csv

Generate recommendations using cached models

Launch a fully functional web interface at http://localhost:8501

### For Developers & Learning
If you want to understand the system architecture step by step, run the files in this order:

Data Preprocessing (if you have raw data):

```bash
python code/data_preprocessing.py
```
Content-Based Model Training:

```bash
python code/content_based.py
```
This generates recommendation tables and feature models in the results/ directory.

Launch Web Interface:

```bash
streamlit run code/main.py
```
Note: The main.py file works independently and doesn't require running content_based.py first—it creates models on-the-fly!

### Collaborative Filtering Module
Important: The collaborative.py file is included for educational purposes only. It demonstrates alternative recommendation approaches but is not integrated into the main application. Future versions may incorporate this as a hybrid recommendation system.


##  Key Features

### Core Recommendation Engine
- **Hybrid Feature Engineering**: Combines TF-IDF text analysis with one-hot encoding of categorical features
- **Content-Based Filtering**: Uses cosine similarity between user profiles and item features
- **Cold-Start Handling**: Creates user profiles from explicit preferences for new users
- **KNN Item-Based**: Additional item similarity recommendations using Nearest Neighbors
- **Unique Item Guarantee**: Ensures no duplicate recommendations

### User Interface
- **Streamlit Web App**: Modern, responsive interface with sidebar navigation
- **New User Registration**: Dynamic form for preference collection
- **Real-time Recommendations**: Instant personalized results
- **Downloadable Outputs**: Export recommendations as CSV

### Technical Features
- **Feature Caching**: Pre-trained models and feature matrices for fast inference
- **Modular Design**: Separate backend processing and frontend interface
- **Comprehensive Outputs**: Generates Top-10, Top-20, and KNN predictions

