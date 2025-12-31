# 🚀 Machine Learning & AI Portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

A comprehensive portfolio showcasing **5 distinct Machine Learning and AI projects**, ranging from Supervised Learning (Regression/Classification) to Unsupervised Learning and NLP. 

The entire portfolio is wrapped in an interactive **Streamlit** web application, allowing users to input real-time data and see predictions instantly.

---

## 🔗 Live Demo
**[👉 Click here to view the Live App](https://portfolio-ml-vj57cd7kcpzsgtilwpaztk.streamlit.app/)** 

---

## 📂 Project Catalog

### 1. 📉 Customer Churn Prediction
* **Problem:** Predict if a bank customer is likely to leave (churn).
* **Type:** Binary Classification.
* **Tech:** XGBoost, SMOTE (for class imbalance), Pandas.
* **Key Feature:** Risk probability score with visual indicators.

### 2. 💰 Real Estate Price Estimator
* **Problem:** Estimate the median house value in California districts based on demographics.
* **Type:** Regression.
* **Tech:** Random Forest Regressor, Scikit-Learn.
* **Key Feature:** Interactive map visualization and dynamic column reordering for inference.

### 3. 🧩 Customer Segmentation
* **Problem:** Group mall customers to identify target marketing audiences.
* **Type:** Unsupervised Learning (Clustering).
* **Tech:** K-Means Clustering, Elbow Method.
* **Key Feature:** Interactive **Plotly** chart visualizing clusters and user position.

### 4. 📅 Sales Forecasting
* **Problem:** Predict future monthly car sales based on historical data.
* **Type:** Time Series Forecasting.
* **Tech:** Prophet (by Meta).
* **Key Feature:** dynamic horizon slider (predict 1 to 24 months ahead).

### 5. 🤖 NLP Sentiment Classifier
* **Problem:** Analyze the emotional tone (Positive/Negative) of text reviews.
* **Type:** Natural Language Processing (NLP).
* **Tech:** TF-IDF Vectorizer, Logistic Regression, WordCloud.
* **Key Feature:** Real-time text analysis with confidence scores and keyword visualization.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Frontend:** Streamlit (Multipage App)
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn
* **Time Series:** Prophet
* **NLP:** NLTK / Scikit-Learn
* **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud
* **Model Management:** Joblib

---

## 💻 Installation & Local Usage

To run this portfolio on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/oscartma/portfolio-ml.git](https://github.com/oscartma/portfolio-ml.git)
cd ml-portfolio
```

### 🚀 Machine Learning & AI Portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

A comprehensive portfolio showcasing **5 distinct Machine Learning and AI projects**, ranging from Supervised Learning (Regression/Classification) to Unsupervised Learning and NLP. 

The entire portfolio is wrapped in an interactive **Streamlit** web application, allowing users to input real-time data and see predictions instantly.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python
* **Frontend:** Streamlit (Multipage App)
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn
* **Time Series:** Prophet
* **NLP:** NLTK / Scikit-Learn
* **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud
* **Model Management:** Joblib

---

## 💻 Installation & Local Usage

To run this portfolio on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/oscartma/portfolio-ml.git](https://github.com/oscartma/portfolio-ml.git)
cd portfolio-ml
```

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train the Models (Crucial Step!)
The app requires pre-trained models (.pkl files) to function. Run the training notebooks to generate them:

# This will create the 'models/' folder and save the .pkl files

### jupyter notebook
 Open and run all cells in:
 - training_churn.ipynb
 - training_prices.ipynb
 - training_segmentation.ipynb
 - training_forecasting.ipynb
 - training_nlp.ipynb

4. Run the App
```bash
streamlit run app.py
```

### 📁 Repository Structure

├── app.py                   # Main entry point (Navigation)
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── projects/                # Streamlit pages for each project
│   ├── 01_churn.py
│   ├── 02_precios.py
│   ├── 03_segmentacion.py
│   ├── 04_forecasting.py
│   └── 05_nlp.py
├── models/                  # Saved .pkl files (Generated by notebooks)
└── notebooks/               # Jupyter Notebooks for training
    ├── training_churn.ipynb
    ├── training_prices.ipynb
    ├── ...
🤝 Contact
Created by Oscar Tibaduiza 💼 LinkedIn: linkedin.com/in/oscartibaduiza

📧 Email: oscartibaduiza@hotmail.com


