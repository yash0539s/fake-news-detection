Fake News Detection using Machine Learning Algorithms

The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not. The distribution of fake news can potentially have highly adverse effects on people and culture. This project involves building and training a model to classify news as fake news or not using a diverse dataset of news articles. We have used four techniques to determine the results of the model.

Logistic Regression
Decision Tree Classifier
Gradient Boost Classifier
Random Forest Classifier
Project Overview
Fake news has become a significant issue in today's digital age, where information spreads rapidly through various online platforms. This project leverages machine learning algorithms to automatically determine the authenticity of news articles, providing a valuable tool to combat misinformation.

Dataset
We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:

True: Genuine news articles
False: Fake or fabricated news articles
System Requirements
Hardware :

4GB RAM
i3 Processor
500MB free space
Software :

Anaconda
Python
Dependencies
Before running the code, make sure you have the following libraries and packages installed:

Python 3
Scikit-learn
Pandas
Numpy
Seaborn
Matplotlib
Regular Expression
You can install these dependencies using pip:

pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn 
pip install re 
Usage
Clone this repository to your local machine:
git clone https://github.com/kapilsinghnegi/Fake-News-Detection.git
Navigate to the project directory:
cd fake-news-detection
Execute the Jupyter Notebook or Python scripts associated with each classifier to train and test the models. For example:
python random_forest_classifier.py
The code will produce evaluation metrics and provide a prediction for whether the given news is true or false based on the trained model.
Results
We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files.
