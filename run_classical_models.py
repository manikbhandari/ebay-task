import argparse
import numpy as np
import pandas as pd
import logging

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from models import sklearn_name2model
from utils import init_logger


def load_data(args):
    train_df = pd.read_csv(args.train_file, header=0, names=['id', 'text', 'target'])
    val_df = pd.read_csv(args.val_file, header=0, names=['id', 'text', 'target'])
    test_df = pd.read_csv(args.test_file, header=0, names=['id', 'text', 'target'])

    # Remove missing values
    train_df = train_df.dropna(axis='rows', how='any', subset=['text'])
    val_df = val_df.dropna(axis='rows', how='any', subset=['text'])
    test_df = test_df.dropna(axis='rows', how='any', subset=['text'])

    X_train, count_vect, tfidf_transformer = featurize_text(train_df)
    X_val, _, _ = featurize_text(val_df, count_vect, tfidf_transformer)
    X_test, _, _ = featurize_text(test_df, count_vect, tfidf_transformer)

    Y_train = train_df['target'].tolist()
    Y_val = val_df['target'].tolist()
    Y_test = test_df['target'].tolist()

    assert (len(Y_train) == X_train.shape[0])
    assert (len(Y_val) == X_val.shape[0])
    assert (len(Y_test) == X_test.shape[0])
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def featurize_text(df, count_vect=None, tfidf_transformer=None):
    X = df['text'].tolist()
    if count_vect is None:
        count_vect = CountVectorizer(ngram_range=(1, 1),
                                     max_df=0.8, min_df=20, max_features=None)
        count_vect.fit(X)

    X_feats = count_vect.transform(X)

    if tfidf_transformer is None:
        tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        tfidf_transformer.fit(X_feats)

    X_tfidf = tfidf_transformer.transform(X_feats)

    return X_tfidf, count_vect, tfidf_transformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='train.log', help="Log file")
    parser.add_argument('--train_file', default='train.csv',
                        help="csv file for training, must be in the format id, text, label")
    parser.add_argument('--val_file', default='valid.csv',
                        help="csv file for validation, must be in the format id, text, label")
    parser.add_argument('--test_file', default='test.csv',
                        help="csv file for testing, must be in the format id, text, label")
    parser.add_argument('--model', default='nb', help="Type of model e.g. svm")
    args = parser.parse_args()

    logger = init_logger(args.log_file, logging.INFO)

    model = sklearn_name2model.get(args.model, None)
    if model is None:
        raise ValueError(f"{args.model} not supported")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(args)
    clf = model().fit(X_train, Y_train)
    val_preds = clf.predict(X_val)
    logger.info(f"Model: {args.model}")
    logger.info(f"validation accuracy: {np.mean(val_preds == Y_val)}")
    test_preds = clf.predict(X_test)
    logger.info(f"test accuracy: {np.mean(test_preds == Y_test)}")

