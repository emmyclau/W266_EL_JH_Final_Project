
# w266_EL_JH_final_project


## Deeper Understanding: Predicting Review Rating with Text ##
- Ching Lau and Joanna Huang

 
#### Abstract

The purpose of this paper is to use user review data to predict sentiment rating and understand how different models are making the prediction. 

We first extract narrative features based on our real world understanding from user reviews for the baseline regression models. Next we build a variations of bag-of-words and CNN models. We use integrated gradients to understand the most important words that each model uses for its prediction and then analyze the sensitivity of these models to different words present in the user reviews.

We find that when a model ignores important information in making its prediction, its model performance is more susceptible to changes in input features.  Therefore, high accuracy is an indicator of a good model only if the model is picking up the right words for the prediction.


#### This Repository

This repository contains all neural network related source codes under https://github.com/emmyclau/W266_EL_JH_Final_Project/tree/master/bow_and_cnn
