# 📰 Fake News Detection

Fake News Detection is a Machine Learning & Deep Learning project that classifies news articles as Real or Fake. Using Natural Language Processing (NLP) techniques and neural network models, this project helps identify misinformation in news content.


## ⚡ Features

🔹 Text preprocessing: tokenization, stemming, stop-word removal

🔹 Word representation: TF-IDF and One-Hot Encoding

🔹 Deep Learning: Bidirectional LSTM with embedding layers

🔹 Machine Learning alternatives: Naive Bayes, Logistic Regression, etc.

🔹 Real-time predictions: input a news headline and content → get Fake or Real

🔹 Model evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix


## 📂 Dataset

* Source: Kaggle: Multimodal Fake News Classification
* Columns: title, text, label
* Split into train, test, and evaluation sets


## 🛠️ Technologies Used

* Python 3
* TensorFlow / Keras (Deep Learning)
* Scikit-learn (Machine Learning)
* NLTK (NLP)
* Pandas & NumPy (Data Handling)

## 🚀 Usage
```
Clone the repository:

git clone https://github.com/username/FakeNewsDetection.git
```
```
Install dependencies:

pip install -r requirements.txt
```

Run the notebook or script to train the model and make predictions.
```
 📝 Example
news = "Scientists confirm that drinking coffee makes you immortal."
prediction = predict_news(news, model)
print(prediction)  # Output: FAKE
```
```
 👤 Author

Priyanshu Samanta

GitHub: https://github.com/PriyanshuSamanta
```
## 🌟 Optional Badges for GitHub

You can also add these to the top of your README for a modern look:

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)
