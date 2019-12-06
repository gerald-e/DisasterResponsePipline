# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)
5. [Instructions](#instructions)

## Project Motivation<a name="motivation"></a>

During real disasters many tweets and messages are sent. These messages are not always unique. In this project, a classification of this natural language is performed. The news results can be sent to potential and appropriate civil protection authorities.

The project is divided into the following sections:

1. Data processing, ie ETL pipeline, to extract data from the source, to cleanse data and to store it in a suitable database structure
2. Machine learning pipeline for training a model to classify text messages into categories
3. Web App to show real time model results.

## Installation <a name="installation"></a>

Python standard configuration
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

The following packages need to be installed for nltk:
* punkt
* wordnet

## File Descriptions <a name="files"></a>

1. data
    - disaster_categories.csv: dataset including all the categories 
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web applicatin
4. root
    - README.md: This file, a description about this project
    - ETL Pipeline Preparation.ipynb: Jupiter notebook for ETL pipeline preparation as a basis for process_data.py
    - ML Pipeline Preparation.ipynb: Jupiter notebook for machine learning pipeline preparation as a basis for train_classifier.py

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project. 

* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/)

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
