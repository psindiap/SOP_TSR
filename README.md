# Fake News Detection

## Overview

The recent proliferation of social media has undoubtedly brought about many benefits,
but along with it also a serious impediment to the society in the form of "Fake News" which has
become an eminent barrier to journalism, freedom of expression, and democracy, as a whole.
Along with differentiating between different types of fake news and misinformation, the study
also aims to understand the extent of social media as a source of information and the impact of
AI-based detection techniques in combating them, with the help a questionnaire involving
university students where we focus on the following dimensions: extent of social media as a
source of news, news verification practices, and frequency of encountering fake news.  

## Project
We analyzed currently used techniques to detect fake news, identify their shortcomings and compare them with the emerging models. We compared the performances of the models in light of
parameters like the words used, the words before and after them, and the overall relationships
between the words in a text. We also compared the models on a dataset based on social media
and compared the changes in performance using Ensemble Learning approaches. The study
aimed to identify suitable models for Fake News Detection. This is in hopes to eventually
promote a safe and healthy environment for sharing information and content online, and in the
process, help develop strategies and techniques to curb the spread of fake news on social media.  

The intended application of the project is for use in applying visibility weights in social media.  Using weights produced by this model, social networks can make stories which are highly likely to be fake news less visible.

## Dataset Description

* train.csv: A full training dataset with the following attributes:
  * id: unique id for a news article
  * title: the title of a news article
  * text: the text of the article; could be incomplete
  * label: a label that marks the article as potentially unreliable
    * 0: unreliable
    * 1: reliable

* test.csv: A testing training dataset with all the same attributes at train.csv without the label.


## More Information
All the relevant information can be found on the pdf in the repository titled **The Role of AI in Combating Fake News.pdf**.
