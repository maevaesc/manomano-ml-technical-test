import numpy as np
import pandas as pd
from process_utils import scaling, process_ingredients, process_non_textual, process_date


def test_standardize():
    x = np.load("tests_refs/standardize/input.npy")
    ref = np.load("tests_refs/standardize/ref.npy")
    print(x.shape)

    assert (scaling(x) == ref).all(), "The output of scaling differs from reference"


def test_process_ingredients():
    input_df = pd.read_csv("tests_refs/process_ingredients/input.csv")
    ref = pd.read_csv("tests_refs/process_ingredients/ref.csv")
    assert (process_ingredients(input_df).to_numpy() == ref.to_numpy()).all(), \
        "The output of ingredients_processing differs from reference"


def test_process_date():
    input_df = pd.read_csv("tests_refs/process_date/input.csv")
    ref = pd.read_csv("tests_refs/process_date/ref.csv")
    assert process_date(input_df, "marketing_authorization_date").equals(ref), \
        "The output of date processing differs from reference"


def test_process_non_textual():
    input_df = pd.read_csv("tests_refs/process_non_textual/input.csv")
    ref = pd.read_csv("tests_refs/process_non_textual/ref.csv")
    assert (process_non_textual(input_df).to_numpy() == ref.to_numpy()).all(), \
        "The output of process_non_textual() differs from reference"
