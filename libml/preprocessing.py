import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_vectorizer():

    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000)


def build_pipeline(vectorizer_text: bool = True):
    
    if vectorizer_text:
        vectorizer = build_vectorizer()
        classifier = LogisticRegression(max_iter=200)
        pipeline = Pipeline([("vectorizer", vectorizer),
                            ("classifier", classifier)])
    else:
        classifier = LogisticRegression(max_iter=200)
        pipeline = Pipeline([("classifier", classifier)])
    return pipeline


def data_preprocessing(
    filepath: str,
    test_size: float = None,
    random_state: int = None,
    vectorizer_output: str = None,
    vectorizer_input: str = None,
):
    """
    Load a TSV dataset with a 'Review' column, clean and stem the text,
    then either fit a new CountVectorizer (and optionally split & save it),
    or load an existing vectorizer to transform the data.

    Args:
        filepath: path to the dataset file (tab-delimited, last column is label).
        test_size: fraction of data to reserve for testing (if splitting).
        random_state: random seed for train/test split.
        vectorizer_output: filepath to save the fitted CountVectorizer (if creating one).
        vectorizer_input: filepath of a pretrained CountVectorizer to load and use.
    """
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    stops = set(stopwords.words("english")) - {"not"}

    # read and clean the reviews
    df = pd.read_csv(filepath, sep="\t", quoting=3)

    def _clean_text(text: str) -> str:
        tokens = re.sub(r"[^a-zA-Z]", " ", text).lower().split()
        return " ".join(stemmer.stem(w) for w in tokens if w not in stops)

    corpus = df["Review"].astype(str).map(_clean_text).tolist()

    # if a pretrained vectorizer is given, just transform and return
    if vectorizer_input:
        with open(vectorizer_input, "rb") as f:
            vec = pickle.load(f)
        return vec.transform(corpus).toarray()

    else:
        vec = CountVectorizer(max_features=1420)
        X = vec.fit_transform(corpus).toarray()
        y = df.iloc[:, -1].values

    # save the fitted vectorizer if requested
    if vectorizer_output:
        with open(vectorizer_output, "wb") as f:
            pickle.dump(vec, f)

    # split into train/test if parameters provided
    if test_size is not None and random_state is not None:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    return X, y


if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(
        project_root, "data", "a1_RestaurantReviews_HistoricDump.tsv"
    )
    vectorizer_output = os.path.join(
        project_root, "output", "c1_BoW_Sentiment_Model.pkl"
    )
    test_size = 0.2
    random_state = 42
    vectorizer_input = None

    # Run preprocessing
    result = data_preprocessing(
        filepath=filepath,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_output,
        vectorizer_input=vectorizer_input,
    )

    # Print results summary
    X_train, X_test, y_train, y_test = result
    print(f"Dataset loaded from: {filepath}")
    print(f"Model will be saved to: {vectorizer_output}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
