# Drug Price Prediction - Maëva Escoulan

## Instruction to run Dockerfile

You can create a new docker image by running this command in the directory where the Dockerfile is located:

``docker build -t image .``

You can then run the image using:

``docker run -it  image``

## Description of the project

The code is located in the `project` folder

This project is a basic machine learning pipeline composed of a train.py and test.py scripts, plus a 
preprocess_utils.py file containing all functions used to process the datasets.

The trained final classifier is saved in 'saved_models/classifier.joblib' file. It can be generated using the 
following command:

``python train.py --train_data ./datasets/drugs_train.csv``

After a few minutes, you should see the following output:

```
Baseline prediction error: 32.49
Mean absolute error at k = 0: 20.11
Mean absolute error at k = 1: 16.91
Mean absolute error at k = 2: 19.58
Mean absolute error at k = 3: 21.89
Mean absolute error at k = 4: 19.23
Final mean absolute error: 19.54
```

To test and generate the submission.csv file, you can run:

``python test.py --classifier_file ./saved_models/classifier.joblib --test_data ./datasets/drugs_test.csv``

The 'saved_models' directory also contains the tfidf transformations fitted on train dataset.

## Unit Tests

Units tests are written in unit_tests.py, they can be run with the following command:
 
``pytest -v unit_tests.py``

They are used to verify the outputs of feature engineering functions

## Comments on results and possible improvements

First, in order to have an idea about my model performances, I first computed a baseline prediction error, that 
is the error I would get with a model that would always predict the mean price of my dataset. 
This baseline error is 32.49

I first processed the drug_train.csv and drug_test.csv files without complex textual information like 'dosage_form'
'route_of_administration' and 'pharmaceutical_companies' that would require specific processing
I did simple processing like casting the 'reimbursement_rate' to values in [0:1], gathering marketing information
that have the same meaning, casting dates values to their year, and one hot encoding of categorical variables

Training a Random Forest Regressor with 1000 estimators and 5-fold cross validation on these features gave poor 
performances, I got a final mean absolute error of 23.86.

I then applied processing on textual information 'dosage_form', 'route_of_administration', 'pharmaceutical_companies',
and 'active_ingredients'. I applied the TfidfTransformer to extract features from these corpus of text.
I noticed that the only information that really had an impact on the performance were the 'dosage_form' and 
"route_of_administration", when adding this to the training features, I got a mean absolute error of 19.53.

Adding the other features with the same Tfidf transformation did not have any impact on my performances. This must be 
due to a bad processing. 
For instance, the active ingredient data are composed words, like 'CHLORHYDRATE DE CIPROFLOXACINE' or 'DICHLORHYDRATE 
DE BÉTAHISTINE', so this feature might be better exploited if using 2-Grams of words for instance.
Plus, the number of columns added with 'pharmaceutical_companies' and 'active_ingredients' is 424 and 1436, which is 
high compared to the number of rows and maybe explains why we the regressor does not learn with the features.

I did not have much time to do further investigations, but here is what I would have liked to explore to get a better 
understanding of the dataset and improve the accuracy:
- Do more data visualisation to get a better insight on the impact of each feature on the price
- Improve features engineering, especially textual features. 
- Identify more precisely features that have an impact on the price
- Reduce the dimensionality of the features (Using the principal component analysis for instance). Indeed, when keeping 
  all the provided information, we have a total of 2046 columns which is too high compared to the number of rows 8564.
- Test different kind of classifiers + parameters tuning. I could have chosen another type of regressor than random forest
but did not have enough time to do more extensive comparisons.
-  Do more extensive grid search for parameters tuning. I only compared my regressor with values [10, 100, 1000] for 
   number of estimators, and finally chose 1000 which was giving better accuracy.




