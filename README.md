# DisasterResponse
Data Scientist Nana Degree Project - Data Engineering

This project is part of the Udacity Data Scientist Nanodegree Program: Disaster Response Pipeline Project and the goal was to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) to build a model for an API that classifies disaster messages.

We will follow machine learning lifecycle process to approach this problem
![addProj](../_attachmnet/CRISPDM.png)

**Business Understanding** Disaster response organizations have to to filter and pull out the most important messages from this huge amount of communications. Organizations do not have enough time to filter out these many messages manually.In this project we will use machine learning to analyze the text messages and communication into different categories like medical supplies, food, or block road.

**Data Provided** We will be analyzing real messages that were sent during disaster events. The data was collected by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) and provided by Udacity. More information can be found [here](https://appen.com/datasets/combined-disaster-response-data/)

Let’s look at the data description:
  1. messages.csv: Contains the id, message that was sent and genre i.e the method (direct, tweet..) the message was sent.
  2. categories.csv: Contains the id and the categories (related, offer, medical assistance..) the message belonged to.

**Data Understanding and ETL** The dataset is provided is basically composed by two files:
disaster_categories.csv: Categories of the messages
disaster_messages.csv: Multilingual disaster response messages

Data preparation steps:
  - Merge the two datasets
  - Split categories into separate category columns
  - One-hot encode category
  - Remove duplicates
  - Upload to SQLite database
