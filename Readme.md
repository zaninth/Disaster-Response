# Disaster-Response-Pipelines

## 🚀**Introduction**
This project is part of The [Udacity](https://eu.udacity.com/) Data Scientist Nanodegree Program. The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. 
The aim of the project was divided into 3 sections:

* **Data Processing**: build an ETL-Pipeline to extract data from the given dataset, clean the data, and then store it in a SQLite database.
* **Machine Learning Pipeline**: creating a ML-pipeline that takes the data from the database and processes text and performs a multi-output 
classification. The script uses NLTK, scikit-learn's Pipeline and GridSearchCV.
* **Web development** deploying the trained model in a Flask web app where you can input a new message and get classification results in 
different categories.

## 📁 **Data/File Description**
* Best Model:
    - Metadata of best model.
    - stages/RandomForestClassifier.
* Images: 
    - All images of Data Analisys.
* Scored json:
    - 4 files off json format with small dataset already scored by the model.
    - sparkify_ranked.ipynb - notebook with process took to ranke the scored dataset.
* Main:
    - Sparkify.ipynb - notebook with all steps since exploratory analisys until finished model.
    - Sparkify_functions.ipynb - nootebook with all functions and pipelines extracted and ready to run on larges datasets. AWS EMR.
    - Sparkify_ranked.ipynb - notebook with process took to ranke the scored dataset.

## ⚠️**Software and Libraries**
* This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [Spark](https://spark.apache.org/docs/3.1.3/api/python/index.html#)
* [Matplotlib](https://matplotlib.org/stable/index.html)
 
## 🚀 References

- [Getting Pandas like dummies in PySpark](https://stackoverflow.com/questions/42805663/e-num-get-dummies-in-pySpark)

- [Using multiple if-else conditions in a list comprehension](https://stackoverflow.com/questions/9987483/elif-in-list-comprehension-conditionals)

- [Business Insider article on classifying region based on U.S. State](https://www.businessinsider.in/The-US-government-clearly-defines-the-Northeast-Midwest-South-and-West-heres-where-your-state-falls/THE-MIDWEST/slideshow/63954185.cms)

- [Write single CSV file (instead of batching) using
  Spark](https://stackoverflow.com/questions/31674530/write-single-csv-file-using-spark-csv)
  
<a id="ref_lr"></a>

- [Python API docs for Logistic Regression](https://spark.apache.org/docs/2.1.1/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression)

<a id="ref_rf"></a>

- [Python API docs for Random Forest Classifier](https://spark.apache.org/docs/2.1.1/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier)

<a id="ref_gbt"></a>

- [Python API docs for GBTClassifier](https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassifier)

<a id="f1_blog"></a>

- [Knowledge about F1 score and why it is a better metric for imbalanced data
  set](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c)

## Acknowledgments
I would like to thank the team from [Udacity's](https://www.udacity.com/) for the great support and the brilliant online 
course [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

## 😖Troubleshoot
Any issues??? Feel free to ask.[Linkedin](https://www.linkedin.com/in/thales-zanin/)

If you find this repo useful,don't forget to give a ⭐
