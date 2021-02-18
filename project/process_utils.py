import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

pd.options.mode.chained_assignment = None  # default='warn'


def scaling(x):
    """Scale features to have values in [0:1]:

    Args:
        x (np.array): input non-scaled features.

    Returns:
        x_scaled (np.array): scaled features.
    """

    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    return x_scaled


def process_ingredients(active_ingredients):
    """Processing of active ingredients data

    Args:
        active_ingredients (pandas.DataFrame): active ingredients data.

    Returns:
        active_ingredients (pandas.DataFrame): active ingredients data with unique drug_id.
    """

    active_ingredients["active_ingredient"] = active_ingredients \
        .groupby(["drug_id"])["active_ingredient"] \
        .transform(lambda x: ' '.join(x))
    active_ingredients = active_ingredients.drop_duplicates()
    return active_ingredients


def process_date(features, column_name):
    """Processing of columns containing date information, keep only year info because month and day are always the same:

    Args:
        features (pandas.DataFrame): the whole features.
        column_name (str): name of the column containing the year information.

    Returns:
        processed_features (pandas.DataFrame): features with new date related columns.
    """

    features[[f'{column_name}_year']] = features[column_name].apply(lambda x: str(x)).apply(lambda x: int(x[0:4]))
    features = features.drop([column_name], axis=1)
    return features


def process_corpus(features, column, train=False):
    """Process textual information with Tfidf method:

    Args:
        features (pandas.DataFrame): all data contained in drug dataframe.
        column (str): name of the column to process
        train (bool): indicates if we want to fit the data

    Returns:
        pipe (sklearn.pipeline): classifier for corpus processing
        corpus (list): corpus of words
    """

    corpus = features[column].tolist()
    if train:
        vocabulary = []
        for doc in corpus:
            vocabulary += doc.strip('.,!;()[]').split()
        vocabulary = list(set(vocabulary))
        stop_words = ['et', 'de', 'à', 'en', 'pour', 'par', "D'", "DE", '.', ')', '(', ',']
        pipe = Pipeline(
            [('count',
              CountVectorizer(vocabulary=vocabulary, stop_words=stop_words)),
             ('tfid', TfidfTransformer())]).fit(corpus)
    else:
        pipe = None
    return pipe, corpus


def process_non_textual(features):
    """Preprocessing of all data in drugs_*.csv file that don't contain complex textual information:

    Args:
        features (pandas.DataFrame): all data contained in drug dataframe.

    Returns:
        processed_features (pandas.DataFrame): processed features.
    """

    # 1: extract columns that don't contain complex textual descriptions
    processed_features = features[['administrative_status',
                                   'marketing_status',
                                   'reimbursement_rate',
                                   'approved_for_hospital_use',
                                   'marketing_authorization_status',
                                   'marketing_authorization_process',
                                   'marketing_declaration_date',
                                   'marketing_authorization_date']]
    # 2: cast reimbursement rate to a value in [0, 1]
    processed_features[['reimbursement_rate']] = \
        processed_features['reimbursement_rate'] \
            .apply(lambda x: str(x)) \
            .apply(lambda x: x.replace("%", "")) \
            .apply(lambda x: float(x) / 100)
    # 3: group marketing related categories that have the same meaning
    processed_features['marketing_status'] = \
        processed_features['marketing_status'].str.replace("Déclaration de suspension de commercialisation",
                                                           "Déclaration d'arrêt de commercialisation")
    processed_features['marketing_authorization_status'] = \
        processed_features['marketing_authorization_status'].str.replace("Autorisation abrogée",
                                                                         "Autorisation retirée")
    processed_features['marketing_authorization_status'] = \
        processed_features['marketing_authorization_status'].str.replace("Autorisation suspendue",
                                                                         "Autorisation retirée")
    # 4: One-hot encoding of categorical features
    processed_features = pd.get_dummies(processed_features)
    # 5: Keep only year in columns containing dates
    processed_features = process_date(processed_features, 'marketing_declaration_date')
    processed_features = process_date(processed_features, 'marketing_authorization_date')
    return processed_features


def process_csv(drugs_data, train=False):
    """Global preprocessing of data in drugs_*.csv files:

    Args:
        drugs_data (pandas.DataFrame): drugs related features.
        train (bool): indicates if labels should be returned or not

    Returns:
        processed_features (pandas.DataFrame): processed features.
        labels (pandas.DataFrame): associated labels if train mode.
    """

    processed_features = process_non_textual(drugs_data)
    if train:
        labels = drugs_data['price']
    else:
        labels = None
    return processed_features, labels
