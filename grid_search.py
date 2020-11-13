import argparse
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from utils import init_logger
from run_classical_models import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='train.log', help="Log file")
    parser.add_argument('--train_file', default='train.csv',
                        help="csv file for training, must be in the format id, text, label")
    parser.add_argument('--val_file', default='valid.csv',
                        help="csv file for validation, must be in the format id, text, label")
    parser.add_argument('--test_file', default='test.csv',
                        help="csv file for testing, must be in the format id, text, label")
    args = parser.parse_args()

    logger = init_logger(args.log_file, logging.INFO)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(args)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True],
        'vect__max_df': [0.6, 0.7, 0.8],
        'vect__min_df': [10, 20, 50],
    }
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=4)
    gs_clf.fit(X_train, Y_train)
    print(gs_clf.best_params_)