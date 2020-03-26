# Disaster Response Pipeline Project
This project provides a web app where an emergency worker can input a new message and get classification results in several categories. The classification model in the backend is trained on real disaster data provided from [Figure Eight](https://www.figure-eight.com/).

## Installation
To run this application Python 3 is required. All needed packages will be imported in each python script.

## Project Motivation
This project is the final exam for the module Data Engineering in the the Udacity Nanodegree [Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Main target was to develop on the data engineering skills to expand opportunities and potential as a data scientist. For that a analyzation of disaster data from Figure Eight was done and a model built for an API that classifies disaster messages.

## File Descriptions
The project is divided in three steps and structured in the following folders. 

##### data/process_data.py
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

##### models/train_classifier.py
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

##### app/run.py
- load data and trained model
- provides web page that handles user query and displays model results

## How to Interact in this project

To run ETL pipeline that cleans data and stores in database
```sh
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
Parameters:
- Filepath messages
- Filepath categories
- Filepath for database to save (existing database will be overwritten)

To run ML pipeline that trains and saves classifier
Arguments:
```
$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
Parameters:
- Filepath to existing database
- Filepath to existing and trained classifier

Run the following command in the app's directory to run the web app.
```
$ python run.py
```

Go to http://0.0.0.0:3001/

#### Screenshots
![](https://github.com/m444x/Disaster-Response-Pipeline/blob/master/disaster-response-project1.png)
![](https://github.com/m444x/Disaster-Response-Pipeline/blob/master/disaster-response-project2.png)
## Licensing
- Udacity for starter code
- Figure Eight for training data
- MIT