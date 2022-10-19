# Subreddit Classifier with Webscraping, NLP and ML

 - [Problem Statement](#Problem-Statement)
 - [Data Sources](#Data-Sources)
 - [Executive Summary](#Executive-Summary)
 - [Notebook Contents](#Notebook-Contents)
 - [Data Dictionary](#Data-Dictionary)
 - [Conclusion & Recommendations](#Conclusion-&-Recommendations)
 

## Problem Statement
In this project, we will be using webscraping, APIs, Natural Language Processing (NLP) and classfication modelling to classify the subreddit posts r/bicycling and r/motorcycles.

We will follow the data science process to answer the classification problem.
1. Define the problem
2. Gather & clean the data
3. Explore the data
4. Model the data
5. Evaluate the model
6. Answer the problem

--- 
## Data Sources
The sources of the data will be from Reddit. We will use [Pushshift's](https://github.com/pushshift/api) API to collect posts from two subreddits.

The first subreddit will be "Bicycling", a popular subreddit with 1.1m members, which welcomes bicyclists of all skill levels including those who don't yet own a bike. It is a popular place to ask questions or get advice on bicycles, or as a forum to organize meetups with other redditors nearby in the area for local rides.

The second subreddit will be "Motorcycles", also another popular subreddit with 1.5m members. Instead of bicycles, the subreddit discusses on topics that are related to motorcycles new and old, including buying and how-to advice.

These two subreddits were chosen as they had a large following, so posts would likely be of a higher quality. Also bicycles and motorcycles are of a similar nature as both are two-wheeled, transportation and sports-related, but not completely indistinguishable.

---
## Executive Summary
**INTRODUCTION**

This project seeks to create a classification model to seperate posts drawn from two fairly similar sub-reddits.

Reddit is a social news, content, and discussions website. Posts are organised according to subject into user-created 'subreddits'. Members submit content (such as images, texts, and links) to subreddits, which can then be voted up ('upvote') or down ('downvote') by other members.

**METHODOLOGY**

The work was done in 3 separate notebooks. The first one focused on webscraping and getting the relevant data from Reddit, before some initial cleaning was done to ensure the quality of the data. The second notebook focused on preprocessing and EDA by using NLP to make sense of the words from the text. The third notebook focused on building and evaluating the models for the problem.

Firstly, data gathering and initial cleaning was performed with one or more of these steps
- Webscraping
- Reading and displaying datasets
- Combining necessary text data together
- Remove null values
- Dropping duplicates
- Feature engineering

Further preprocessing and EDA was then done after the initial cleaning to get a final dataset
- Create stop word list
- Create stemmed words using Porter Stemmer
- Create lemmatized words using WordNetLemmatizer
- Draw WordCloud to visualise
- Explore word and character length of posts
- Finding most common words in the original text
- Using n-grams to find common phrases

In the last notebook, train-test split was done on the final data set. The main text, stemmed text and lemmatized text were each passed into the 4 models: CountVectorizer with Multinomial Naive Bayes, TfidfVectorizer with Multinomial Naive Bayes, CountVectorizer with Random Forest Classifier and TfidfVectorizer with Random Forest Classifier. The performance of the models were then compared against each other using various metrics discussed below.

**SIGNIFICANT FINDINGS**

The models were tuned using RandomizedSearchCV to save time to get a good working model. Only results from the models with the stemmed text are shown in the table below.

|              | train_score | test_score | generalisation | precision | f1_score | roc_auc_score |
|-------------:|------------:|-----------:|---------------:|----------:|---------:|--------------:|
| cvec_nb_stem |      0.9270 |     0.8773 |         5.3614 |    0.8467 |   0.8712 |        0.9472 |
| tvec_nb_stem |      0.9410 |     0.8773 |         6.7694 |    0.8568 |   0.8694 |        0.9503 |
| cvec_rf_stem |      0.9968 |     0.8442 |        15.3090 |    0.8365 |   0.8303 |        0.9176 |
| tvec_rf_stem |      0.9957 |     0.8529 |        14.3417 |    0.8569 |   0.8374 |        0.9199 |

We will select the **CountVectorizer with the Multinomial Naive Bayes model** as it has good generalisation, accuracy, precision, f1 scores and ROC-AUC scores, compared to the other models. We will apply this model after Porter Stemming, whose overall performance is better than when on its base form or after lemmatizing.

The Random Forest models show signs of overfitting and further work would have to be done to optimise and tune the hyperparameters. The accuracy, precision and f1 scores are also not as high as when done with the Naive Bayes models.


### Notebook Contents
1. [Problem Statement](#Problem-Statement)
2. [Import Libraries](#Import-Libraries)
3. [Data Dictionary](#Data-Dictionary)
4. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
5. [Get Dummies on Categorical columns](#Get-Dummies)
6. [Model Preparation and Fitting](#Model-Preparation)
---

## Data Dictionary

The data comes from two subreddits: Bicycling and Motorcycles. 3000 submissions were pulled from each subreddit and later used for analysis in the project.

| Feature   | Type   | Description                                                                                                                                                  |
|-----------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| subreddit | object | The subreddit where the post was taken from. This is either 'bicycling' or 'motorcycles'.                                                                    |
| text      | object | This is the combined text that was extracted directly from the subreddit's post selftext and title columns. Duplicates and special texts have been removed.  |
| stem_text | object | This column contains the Porter Stemmed text sentences that have been stemmed from the original clean tokens.                                                |
| lem_text  | object | This column contains the lemmatized text sentences that have been lemmatized from the original clean tokens.                                                 |
| bike      | int64  | This is a binary column that indicates which subreddit the post belongs to. 0 for bicycling and 1 for motorcycles.                                           |


---
## Conclusion & Recommendations 
Overall, we are able to get quite a good working model with over 85% accuracy. This project helped me to walk through the steps of how to webscrap directly from the website, analyze the words from the texts, to applying the transformation and classifier models needed to generate a good accuracy score. Many overlapping concepts were used in this project, including using NLP tools, web APIs, classifier models and evaluation techniques like using the ROC-AUC curve and confusion matrix. However, if given more time and data to answer the problem, my recommendations would be to:

- There were still some spam posts left over, so more thorough cleaning would have to be done. Some posts also had weird string characters left despite one round of cleaning.
- Possibly test out on other models other than Naive Bayes and Random Forest.
- We could also explore new features within Reddit such as the upvotes and downvotes, as well as post comments.
- As this is a fairly balanced dataset, the metrics are easy to compare. In the future, if we were to encounter highly imbalanced dateset, the use of sensitivity and specificity would be of greater use. The ROC-AUC curve cannot be used and would have to be substituted with the Precision Recall curve.