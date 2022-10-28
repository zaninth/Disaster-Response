# Disaster-Response-Pipelines

## 🚀**Introduction**
This project is part of The [Udacity](https://eu.udacity.com/) Data Scientist Nanodegree Program.The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. 
The aim of the project was divided into 3 sections:

* **Data Processing**: build an ETL-Pipeline to extract data from the given dataset, clean the data, and then store it in a SQLite database.
* **Machine Learning Pipeline**: creating a ML-pipeline that takes the data from the database and processes text and performs a multi-output 
classification. The script uses NLTK, scikit-learn's Pipeline and GridSearchCV.
* **Web development** deploying the trained model in a Flask web app where you can input a new message and get classification results in 
different categories.

## 📁 **Data/File Description**
* App:
    - Templates with html's archives.
    - run.py - Script to run the web app.
* Preparation: 
    - ETL Pipeline prearation - Jupyter notebook with all data analyses e preparations.
    - ML Pipeline preparation - Jupyter notebook with all ML models and parameters to find the one who best fit to the proposal.
* Data:
    - disaster_categories.csv - Contains the id, message that was sent and genre.
    - disaster_messages.csv - Contains the id and the categories (related, offer, medical assistance..) the message belonged to.
    - process_data.py - Script used for data cleaning and pre-processing.
    - DisasterResponse.db - Database contain cleaned data and load in SQL.
* Models:
    - train_classifier.py - Script used to train the model.
* Images:
    - Images from the web app running.

## ⚠️**Software and Libraries**
* This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [NLTK](https://www.nltk.org/)
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Sqlalchemy](https://www.sqlalchemy.org/)
* [Dash](https://plot.ly/dash/)
* [Flask](https://flask.palletsprojects.com/en/2.2.x/)
 
## 🚀Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/zaninth/Disaster-Response/blob/main/images/distribution%20image.png)
![alt text](https://github.com/zaninth/Disaster-Response/blob/main/images/porportion%20of%20messages.png)
![alt text](https://github.com/zaninth/Disaster-Response/blob/main/images/heatmap.png)
![alt text](https://github.com/zaninth/Disaster-Response/blob/main/images/image%201.png)

## Acknowledgments
I would like to thank the team from [Udacity's](https://www.udacity.com/) for the great support and the brilliant online 
course [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

## 😖Troubleshoot
Any issues??? Feel free to ask.[Linkedin](https://www.linkedin.com/in/thales-zanin/)

If you find this repo useful,don't forget to give a ⭐
