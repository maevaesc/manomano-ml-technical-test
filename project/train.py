import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from process_utils import scaling, process_ingredients, process_corpus, process_csv

parser = argparse.ArgumentParser()
parser.add_argument("--train_data",
                    help="Location of the train data file")
parser.add_argument("--save_path",
                    default="saved_models",
                    help="Location of the saved models")
parser.add_argument("--ingredients_data",
                    default=None,
                    help="Location of the ingredients file")
parser.add_argument("--companies",
                    default=False,
                    help="Add the pharmaceutical companies data for predictions")
args = parser.parse_args()


def grid_search(x, y):
    """grid search to estimate our model's best parameters:

    Args:
        x (np.array): The train features.
        y (np.array): The train labels.

    Returns:
        regressor (sklearn model): The final regressor.
    """
    parameters = {'n_estimators': [10, 100, 1000]}
    regressor = RandomForestRegressor()
    search = GridSearchCV(regressor, parameters, n_jobs=4, verbose=3)
    search.fit(x, y)
    print(search.cv_results_)
    print(search.best_estimator_)
    return search.best_estimator_


def k_fold_training(x, y, n_estimators, k=5):
    """k fold cross validation on the training set:

    Args:
        x (np.array): The train features.
        y (np.array): The train labels.
        n_estimators (int): number of random forest estimator
        k (int): number of folds

    Returns:
        classifier (sklearn model): The final classifier.
    """

    classifier = RandomForestRegressor(n_estimators=n_estimators)
    kf = KFold(n_splits=k)
    total_mean = 0
    for (train, test), i in zip(kf.split(x), np.arange(0, k)):
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        # Scale and fit train data
        x_train_scaled = scaling(x_train)
        classifier.fit(x_train_scaled, y_train)
        # Scale and predict test data
        x_test_scaled = scaling(x_test)
        predictions = classifier.predict(x_test_scaled)
        # Compute accuracy metric
        errors = abs(predictions - y_test)
        print(f'Mean absolute error at k = {i}: {round(np.mean(errors), 2)}')
        total_mean += np.mean(errors)
    print(f'Final mean absolute error: {round(total_mean / k, 2)}')
    classifier.fit(x, y)
    return classifier


def train(train_data, save_path="saved_models", ingredients_data=None, companies=False):
    """Training pipeline:

    Args:
        train_data (str): Location of train data csv.
        save_path (str): location of saved models.
        ingredients_data (str): active_ingredients information, defaults to None if we do not want to use it.
        companies (bool): whether to use pharmaceutical companies information or not

    """

    # Data pre-processing
    base_features = pd.read_csv(train_data)
    features_train, labels_train = process_csv(base_features, train=True)
    pipe_dosage, corpus_dosage = process_corpus(base_features, "dosage_form", train=True)
    pipe_route, corpus_route = process_corpus(base_features, "route_of_administration", train=True)
    # Save the fitted transformation to apply it later on test data
    dump(pipe_dosage, f'{save_path}/pipe_dosage.joblib')
    dump(pipe_route, f'{save_path}/pipe_route.joblib')
    features_train = np.concatenate((pipe_dosage.transform(corpus_dosage).toarray(),
                                     pipe_route.transform(corpus_route).toarray(),
                                     np.array(features_train)), axis=1)
    if companies:
        pipe_companies, corpus_companies = process_corpus(base_features, "pharmaceutical_companies", train=True)
        features_train = np.concatenate((pipe_companies.transform(corpus_companies).toarray(),
                                         features_train), axis=1)
        dump(pipe_companies, f'{save_path}/pipe_companies.joblib')
    if ingredients_data is not None:
        active_ingredients = pd.read_csv(ingredients_data)
        active_ingredients = process_ingredients(active_ingredients)
        merged_features = pd.merge(base_features, active_ingredients, on='drug_id', how='inner')
        pipe_ingredients, corpus_ingredients = process_corpus(merged_features, "active_ingredient", train=True)
        features_train = np.concatenate((pipe_ingredients.transform(corpus_ingredients).toarray(),
                                         features_train), axis=1)
        dump(pipe_ingredients, f'{save_path}/pipe_ingredients.joblib')

    labels_train = np.array(labels_train)

    # Start training
    baseline_errors = abs(np.mean(labels_train) - labels_train)
    print(f'Baseline prediction error: {round(np.mean(baseline_errors), 2)}')
    final_classifier = k_fold_training(features_train, labels_train, 1000)
    dump(final_classifier, 'saved_models/classifier.joblib')


if __name__ == '__main__':
    train(args.train_data, args.save_path, args.ingredients_data, args.companies)
