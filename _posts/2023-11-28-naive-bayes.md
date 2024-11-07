---
layout: post
title: Naive Bayes
date: 2023-11-28 16:40:16
description: Implementing Naive Bayes model to understand social disparity in impacts of climate disasters
tags: climate-justice-project code machine-learning
categories: 
---


## Overview

Naive Bayes is a supervised learning approach for probabilistic classification, applying Bayes’ theorem to assign class labels to vectors of features. Like other supervised learning algorithms, naive Bayes models are trained with labeled data in order to predict values of a target attribute for new data vectors. It makes a (naive) assumption of independence between features (attributes). Naive Bayes computes in linear time, making it highly scalable and suitable for classifying large datasets and performs well if the independence condition holds. Naive Bayes is based on Bayes’ theorem, which describes the probability of an event based on prior knowledge of conditions.

Naive Bayes can be used for multi-class prediction, and is implemented in applications such as text classification, spam filtering, sentiment analysis, and recommendation systems. There are various types of naive Bayes classifiers. Multinomial naive Bayes (MNB) is used for data vectors with features that have two or more variables (for example, a marital status feature with values of married, single and divorced). In MNB, the data features are *multinomially distributed*, which means they represent probability distributions for events that have two or more possible outcomes. The decision rule for multinomial naive Bayes is:


<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/naive-bayes.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    (Image Credit: <a href="https://uc-r.github.io/naive_bayes">UC Business Analytics</a>)
</div>

where *c* is a data label and ***X*** is a data vector. 
- P(*c*|*x*): posterior probability - probability of the class given the predictor attribute vector x
- P(*x*|*c*): likelihood – probability of the attribute vector given the class 
- P(*c*): class prior probability – probability of observing the class
- P(*x*): predictor prior probability – probability of observing the attribute vector

Bernoulli naive Bayes is another naive Bayes classifier where the features used to predict the class are Boolean values (i.e. yes or no), for example, whether a word occurs in the text or not. Whereas multinomial naive Bayes is concerned about the frequency of a variable occurring, Bernoulli naive Bayes is only concerned with whether the variable occurs or not. Furthermore, unlike multinomial naive Bayes, Bernoulli naive Bayes explicitly models the non-occurrence of a variable by penalizing it. Bernoulli naive Bayes is often applied for classifying shorter texts.


<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/naive-bayes-plot.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    (Image Credit: <a href="https://www.analyticsvidhya.com/blog/2022/03/building-naive-bayes-classifier-from-scratch-to-perform-sentiment-analysis/">Koushiki</a>)
</div>

Smoothing is often used in naive Bayes models in order to correct probability estimates of zero which occur if a given class and feature never occur together in the training data. These zero probabilities are problematic because they eliminate the information in other probabilities when probabilities are multiplied during naive Bayes computations. Laplace or Lidstone smoothing methods introduce a small-sample correction so that no probability is exactly zero. 

In this work, I apply naive Bayes to classify whether individuals are able to recover from hurricane impacts a year later. I use a dataset of the Kaiser Family Foundation/Episcopal Health Foundation Poll: Harvey Anniversary Survey, which has survey data on how individuals have been impacted by Hurricane Harvey in 2017 reported 1 year after the storm. I included data features of storm impacts (home damage and reduced work hours) and demographics (race, gender, and income) to predict whether a respondent reported that their day to day life is largely back to normal or still disrupted 1 year later. Such a model can be used to determine the best way to allocate resources in climate relief so that recovery aid can reach those who need it the most. In this work, I incorporate demographics such as race, gender, and income into the climate impacts model to take into account and understand how these socioeconomic vulnerabilities affect an individual’s ability to recover after a climate disaster.

## Data Preparation

As is the case with supervised learning algorithms, Naive Bayes requires labeled data. I used the Harvey Anniversary Survey dataset, which has many different features corresponding to survey questions. A sample of the raw data is shown below. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/harvey-anniversary-raw.png" class="img-fluid" %}
    </div>
</div>
<!-- <div class="caption">
    (Image Credit: <a href="https://www.analyticsvidhya.com/blog/2022/03/building-naive-bayes-classifier-from-scratch-to-perform-sentiment-analysis/">Koushiki</a>)
</div> -->

From the Harvey Anniversary Survey dataset, I was interested in predicting individuals’ abilities to recover from the hurricane. Thus, I chose the attribute corresponding to hurricane recovery as the label for the data. For the predictors, I was interested in a combination of storm impacts and demographics, and chose the following features: whether the respondent sustained home damage as a result of Hurricane Harvey, whether the respondent had hours cut back at work as a result of Hurricane Harvey, as well as the respondent’s race, gender, and income. 

For the hurricane recovery attribute I used as the label, the survey question asks “Which of the following best describes your personal situation in terms of recovering from Hurricane Harvey?” and there are 4 possible outcomes: largely back to normal, almost back to normal, still somewhat disrupted, and still very disrupted. In supervised learning, it is important to make sure the data is balanced - that there are similar numbers of samples for each value of the label, as well as similar numbers of samples for each value of the features. The responses for the recovery label were unbalanced, with the counts of each response from the raw data shown below:


<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/label-balance.png" class="img-fluid" %}
    </div>
</div>

In order to balance the data while retaining as many of the “somewhat disrupted” and “very disrupted” data vectors as possible, I combined the responses into two labels when preparing the data. I titled the label “recovery” and the classes “yes” and “no,” with “yes” including “largely/almost back to normal” responses and “no” corresponding to “still somewhat disrupted” and “still very disrupted” responses. 

Although there were imbalances to varying extents for each attribute, since my project focuses on the social impacts of climate disasters, I prioritized balancing the demographic attributes so that the resulting model isn’t biased to make more accurate predictions for more highly represented identities. Among these attributes, the race attribute was the most unbalanced, with counts from raw data shown below. 

<div class="row">
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/race.png" class="img-fluid" %}
    </div>
</div>

Since there were relatively few respondents that identified as Hispanic, mixed race, or Asian, I omitted these races when cleaning the data. This decision stemmed from my goal to keep more data samples for white and black/African-American races when balancing the data in an attempt to improve prediction using those variables; however, the trade-off is that this model is limited to apply to only those two races. In general, this is a challenge of applying machine learning with minority identities, since large numbers of samples or an oversample of the minorities are needed. 

After preparing and balancing the recovery label and the race attribute, the final balance of the data is as follows: 

<div class="row">
    <div class="col-sm-4 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/final-balance.png" class="img-fluid" %}
    </div>
</div>

A sample of the cleaned data is shown below, and the prepared data can be found [here](https://drive.google.com/file/d/1LS6TzQuZvnKweTjBZRqtCBx5y3VjzA0Z/view).

<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/prepped-data-sample.png" class="img-fluid" %}
    </div>
</div>

For supervised learning algorithms such as Naive Bayes, we need to split the data into a training set to train the model and a testing set to test the accuracy of the model. The training and testing sets must be disjoint, so that the model does not see the testing data during training. This ensures that the testing data can be used to accurately evaluate the model’s performance for making predictions with new data. To prepare the data for Naive Bayes, I used a 3-to-1 training-to-testing data split (i.e. I sampled 75% of the data to use as training data). 

In addition, I also applied Naive Bayes with a k-fold cross validation, a resampling method that divides the data into k parts and uses different portions of the data to iteratively train and test a model. For my model, I used 10-fold cross validation, which tunes the model over 10 iterations, using 9 of the 10 folds for training and 1 fold for testing.


## Code
Code for Naive Bayes in R can be found [here](https://github.com/ruojiasun/machine-learning-project-fa23/blob/master/nb.R). R was used since R can perform Naive Bayes with categorical data.

Code to prepare the data can be found [here](https://github.com/ruojiasun/machine-learning-project-fa23/blob/master/prepare-data-nb-dt.ipynb).

## Results

When modeling an individual’s ability to recover from hurricane impacts using Naive Bayes, the resulting model without cross-validation resulted in an accuracy of 71.43%, averaged over 5 models. A confusion matrix from one of the models is shown below, which indicates that the model has similar rates of correctly predicting the “yes” and “no” recovery labels. The confusion matrix is shown below:

<div class="row">
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/conf-mat.png" class="img-fluid" %}
    </div>
</div>


I also ran a Naive Bayes with a 10-fold cross-validation, which yielded an accuracy of 70.79%, averaged over 5 models. The Naive Bayes algorithm performed similarly with and without the cross-validation. (My implementation of the cross-validation was based on the caret and klaR packages, which did not enable the generation of confusion matrices for each iteration.)

The Naive Bayes with cross-validation indicated that the importance of the variables were as follows: with home damage being by far the most important, followed by income and race. This is significant because the social dimensions of income and race are even better predictors of an individual’s ability to recover after a storm compared to experiencing reduced hours at work due to the storm. On the other hand, the gender attribute had low importance in predicting the recovery label. 


<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/var-imp-nb.png" class="img-fluid" %}
    </div>
</div>

Below, I plotted the conditional probabilities from the Naive Bayes model without cross-validation for each attribute-label pair.

<div class="row">
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/p-home-damage.png" class="img-fluid" %}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/p-reduced-hours.png" class="img-fluid" %}
    </div>
</div>
<div class="row">
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/p-gender.png" class="img-fluid" %}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0  mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/p-race.png" class="img-fluid" %}
    </div>
</div>
<div class="row">
    <div class="col-sm-5 mt-3 mt-md-0 mx-auto d-block">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-11-28-naive-bayes/p-income.png" class="img-fluid" %}
    </div>
</div>

These visualizations further demonstrate that home damage is a strong predictor of lack of recovery, with a 0.8385 conditional probability. These visualizations also show that the demographic attributes and storm recovery are dependent. For example, with the income attribute, the conditional probability of not recovering given an individual is living below the poverty line is about 50% higher than the conditional probability of not recovering given an individual is living above the poverty line.

## Conclusion
I applied Naive Bayes to predict an individual’s ability to recover from hurricane impacts 1 year after the event, using record data from the Harvey Anniversary Survey. I used the predictors of  home damage, reduced work hours, and respondent’s race, gender, and income. The resulting model had an average accuracy of  71.43% without cross validation and 70.79% with 10-fold cross validation. While this accuracy is better than randomly guessing, it is not very high, indicating that the model can perhaps be improved with a larger dataset or by using different features. Alternatively, it could be the case that there are more complex patterns in the data that can be better captured using a different model. 

Applying the Naive Bayes algorithm was able to shed light onto which features and variables were most important in predicting an individual’s ability to recover after a life-altering climate event. While it makes sense that whether an individual sustained home damage due to the hurricane is the most important indicator for their ability to recover, the income and race predictors were the next most important, even more so than experiencing reduced work hours due to the storm. For example, the conditional probability of recovering is 1.5 as high given someone is white compared to if they are African American, and the conditional probability of not recovering is 1.5x as high given someone is living in poverty compared to not living in poverty. These results suggest that it is important to consider socioeconomic vulnerabilities when understanding and predicting the impacts of climate disasters.


<!-- Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. [Pinterest](https://www.pinterest.com) DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy.

#### Hipster list

- brunch
- fixie
- raybans
- messenger bag

#### Check List

- [x] Brush Teeth
- [ ] Put on socks
  - [x] Put on left sock
  - [ ] Put on right sock
- [x] Go to school

Hoodie Thundercats retro, tote bag 8-bit Godard craft beer gastropub. Truffaut Tumblr taxidermy, raw denim Kickstarter sartorial dreamcatcher. Quinoa chambray slow-carb salvia readymade, bicycle rights 90's yr typewriter selfies letterpress cardigan vegan.

<hr>

Pug heirloom High Life vinyl swag, single-origin coffee four dollar toast taxidermy reprehenderit fap distillery master cleanse locavore. Est anim sapiente leggings Brooklyn ea. Thundercats locavore excepteur veniam eiusmod. Raw denim Truffaut Schlitz, migas sapiente Portland VHS twee Bushwick Marfa typewriter retro id keytar.

> We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
> —Anais Nin

Fap aliqua qui, scenester pug Echo Park polaroid irony shabby chic ex cardigan church-key Odd Future accusamus. Blog stumptown sartorial squid, gastropub duis aesthetic Truffaut vero. Pinterest tilde twee, odio mumblecore jean shorts lumbersexual. -->
