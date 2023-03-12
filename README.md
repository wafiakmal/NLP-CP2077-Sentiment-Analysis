# NLP-CP2077-Sentiment-Analysis

# Abstract 

For our group assignment, we had to find a dataset on which we would apply a generative
model and a neural network model. Our dataset of interest was a set of the most recent 16,599
reviews on Cyberpunk 2077, which we scraped from the Steam game store on December 8, 2022,
by using the Steam API. As for how to scrape the data, we obtained the code from a Medium
article regarding review scraping from Steam, (Muller, 2021). Then we conducted data cleaning
on the dataset, which was then reduced to 16,460 reviews, classified as 10,964 recommended
reviews and 5,496 not recommended reviews. Since our reviews have labels of “Recommended”
and “Not Recommended”, we decided to construct a generative model based on an approach
combining bag of words with the Naive Bayes theorem of conditional probability (Ersoy, 2021)
and apply a discriminative model consisting of a bidirectional LSTM neural network model to
predict whether a review has a positive sentiment (recommended) or a negative sentiment (not
recommended). Our models would be trained on a training set of 67% of the data and tested on
33% of the data, and both of them would later be trained and tested in the same manner on a
synthetic dataset created by the generative model.

As for the resulting accuracies, the generative model and the BiLSTM model had overall
accuracy scores of 82.36% and 92.2% on a specific run respectively in predicting the labels of the
original data. For the BiLSTM, it is important to note that the accuracy fluctuates per run, often
oscillating between 80% and 95% after 5 epochs with an embedding size of 64. In the run with
92.2%, it surpassed the generative’s model predictive performance during the third epoch.
However, when training on the synthetic data, an advantage of the generative model was made
apparent. This model achieved 97.20% accuracy at predicting the labels on the training set of the
synthetic data. As for the BiLSTM, it achieved 92.8% and did not surpass the generative model’s
performance. We theorize the main cause behind the performance disparity may have occurred
because the BiLSTM considers order and long range dependencies in a bidirectional manner
(Dolphin, 2020), whereas our bag of words used a Naive Bayes approach when classifying reviews
as well as synthetically generating them. This implies that every review in the synthetic data was
a bag of words with no regard to order, much less the concept of a long range, contextual
dependency. In addition, during the synthetic generation process, each review used tokens
exclusively found within the tokens of its recommended or not recommended class, and there
was a unique token difference of 7,000 tokens between the two classes. In this sense, since our
synthetic reviews disregard order, long-range, contextual dependencies and contained words
obtained randomly from a corpus of a specific class, our generative model was well suited to
applying the Naive Bayes theorem to classify the reviews in the test set and gained an advantage
over the BiLSTM’s affinity for order and handling of long-term memory.


# How did our data look?

![image](https://user-images.githubusercontent.com/70504872/212144615-468609b6-11b8-4704-8f93-185d4c304e66.png)


# Generative Model and LSTM model results for real data

Generative:
![image](https://user-images.githubusercontent.com/70504872/212144783-f57acc5a-fee3-42a4-bd62-022ead97ce20.png)

BiLSTM:
![image](https://user-images.githubusercontent.com/70504872/212145362-2f44b75a-131b-4ae8-9200-a2edc13677cf.png)


# Generative Model and LSTM model results for synthetic data

Generative:
![image](https://user-images.githubusercontent.com/70504872/212145137-eefdcbdb-6423-4822-9948-3ac3c2501f60.png)

BiLSTM:
![image](https://user-images.githubusercontent.com/70504872/212145472-1b8cb59b-748d-436c-a8c2-30b93161fec5.png)


# Reproduceability

All the necessary files to reproduce the project are contained within the G_submission folder.

Steps:

1. Run the steam scraper (A_steam_scraper.py) (This step was completed, but if run again will generate a new dataset, it won't be the same as ours, since the scraper gets the most recent 16K reviews)
2. Master dataset created (cp2077_reviews.csv.zip)
3. Run the cleaning file for generative model (B_cleaning_gen_model.py)
4. Cleaned REAL dataset for the generative model created (cleaned_real_reviews.csv)
## Real Dataset
5. Run the generative model (C_Generative_Model.py) on the cleaned real review (cleaned_real_reviews.csv), we will get the accuracy result of generative model in print statement on command line and a csv file with all the new columns (test_with_new_columns.csv). If we opted to make the synthetic data, we will get a csv for the synthetic data (synthetic_reviews_all_trial_1.csv), but running this file will create a different result each time, because it is a stochastic generation of text. 
6. Run the Bi-LSTM model (lstm_new.ipynb) on the real data, as it has special cleaning (cp2077_reviews.csv.zip), path changing is maybe needed.
## Synthetic Dataset
7. Run the generative model (C_Generative_Model.py) on the synthetic review file (synthetic_reviews_all_trial_1.csv)
8. Run the Bi-LSTM model (lstm_new_synthetic.ipynb) on the synthetic review file (synthetic_reviews_all_trial_1.csv), path changing is maybe needed.
