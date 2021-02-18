import argparse
from joblib import load
import numpy as np
import pandas as pd
from process_utils import scaling, process_ingredients, process_corpus, process_csv

parser = argparse.ArgumentParser()
parser.add_argument("--classifier_file",
                    help="Location and name of the trained classifier")
parser.add_argument("--test_data",
                    help="Location and name of the test data file")
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


def test(classifier_file, test_data, save_path="saved_models", ingredients_data=None, companies=False):
    """Test trained regressor:

    Args:
        test_data (str): Location of test data csv.
        ingredients_data (str): active_ingredients information, defaults to None if we do not want to use it.
        companies (bool): whether to use pharmaceutical companies information or not

    """
    # Apply the same processing as in train data
    base_features = pd.read_csv(test_data)
    features_test, _ = process_csv(base_features)
    _, corpus_dosage = process_corpus(base_features, "dosage_form")
    pipe_dosage = load(f'{save_path}/pipe_dosage.joblib')
    _, corpus_route = process_corpus(base_features, "route_of_administration")
    pipe_route = load(f'{save_path}/pipe_route.joblib')
    features_test = np.concatenate((pipe_dosage.transform(corpus_dosage).toarray(),
                                    pipe_route.transform(corpus_route).toarray(),
                                    np.array(features_test)), axis=1)

    if companies:
        _, corpus_companies = process_corpus(base_features, "pharmaceutical_companies")
        pipe_companies = load(f'{save_path}/pipe_companies.joblib')
        features_test = np.concatenate((pipe_companies.transform(corpus_companies).toarray(),
                                        features_test), axis=1)
    if ingredients_data is not None:
        active_ingredients = pd.read_csv(ingredients_data)
        active_ingredients = process_ingredients(active_ingredients)
        merged_features = pd.merge(base_features, active_ingredients, on='drug_id', how='inner')
        _, corpus_ingredients = process_corpus(merged_features, "active_ingredient")
        pipe_ingredients = load(f'{save_path}/pipe_ingredients.joblib')
        features_test = np.concatenate((pipe_ingredients.transform(corpus_ingredients).toarray(),
                                        features_test), axis=1)

    features_test = scaling(features_test)
    classifier = load(classifier_file)
    predictions = classifier.predict(features_test)

    submission = base_features["drug_id"]
    prices = pd.DataFrame(predictions, columns=['prices'])
    submission = pd.concat([submission, prices], axis=1)
    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    test(args.classifier_file, args.test_data, args.save_path, args.ingredients_data)
