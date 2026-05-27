Fake News Detection System

A Machine Learning-based Fake News Detection System that classifies news articles as **TRUE** or **FAKE** using Natural Language Processing (NLP) and Logistic Regression.



  Project Overview

This project detects whether a news article is real or fake by training a machine learning model on labeled datasets containing real and fake news articles.

The system uses:

* **Python**
* **Pandas**
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Logistic Regression**



  Features

* Detects fake and real news articles
* Uses NLP preprocessing techniques
* TF-IDF vectorization for text analysis
* Logistic Regression classification model
* High accuracy prediction
* Simple and beginner-friendly implementation



  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Pickle



  Dataset

The project uses two datasets:

* `True.csv` → Real news articles
* `Fake.csv` → Fake news articles

Dataset columns:

* `title`
* `text`
* `subject`
* `date`

The datasets are combined and labeled:

* TRUE → `1`
* FAKE → `0`



  Working Process

1. Load True and Fake datasets
2. Add labels to datasets
3. Combine datasets
4. Perform text preprocessing
5. Convert text into numerical vectors using TF-IDF
6. Train Logistic Regression model
7. Predict whether news is TRUE or FAKE



  Installation

Clone the repository:

bash
git clone https://github.com/your-username/fake-news-detector.git


Move into the project folder:

bash
cd fake-news-detector


Install dependencies:

bash
pip install pandas scikit-learn numpy




  Run the Project

bash
python detect.py




  Model Used

* Logistic Regression



 Machine Learning Concepts Used

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Train-Test Split
* Text Classification
* Supervised Learning



  Project Structure

text
fake-news-detector/
│
├── detect.py
├── True.csv
├── Fake.csv
├── model.pkl
├── vectorizer.pkl
└── README.md




  Future Improvements

* Add Deep Learning models (LSTM/BERT)
* Build Flask/Streamlit web app
* Add live news URL checking
* Improve preprocessing and accuracy
* Deploy on cloud platform



 👨‍💻 Author
Yashwanth

