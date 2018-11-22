import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from numpy import asarray

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic_score = None
        min_model = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                d = model.n_features
                p = n ** 2 + 2 * n * d - 1

                bic_score = (-2 * logL + p * math.log(len(self.sequences)))

                if min_bic_score is None or min_bic_score > bic_score:
                    min_bic_score = bic_score
                    min_model = model
            except:
                pass

        return min_model


class SelectorDIC(ModelSelector):

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic_score = None
        max_model = None

        rest_words = list(self.words)
        rest_words.remove(self.this_word)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)

                all_score = 0.0

                for w in rest_words:
                    X, lengths = self.hwords[w]

                    all_score = all_score+model.score(X, lengths)

                dic_score =  score - (all_score / (len(self.words) - 1))

                if max_dic_score is None or max_dic_score < dic_score:
                    max_dic_score = dic_score
                    max_model = model
            except:
                    pass

        return max_model


class SelectorCV(ModelSelector):

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = None
        max_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                all_score = 0.0
                qty = 0
                final_model = None

                if (len(self.sequences) >= 2):
                    folds = min(len(self.sequences),3)
                    split_method = KFold(shuffle=True, n_splits=folds)
                    parts = split_method.split(self.sequences)

                    for cv_train_idx, cv_test_idx in parts:
                        X_train, lengths_train = asarray(combine_sequences(cv_train_idx, self.sequences))

                        X_test, lengths_test = asarray(combine_sequences(cv_test_idx, self.sequences))

                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                        all_score = all_score+model.score(X_test,lengths_test)

                        qty = qty+1
                    score = all_score / qty

                else:
                    final_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    score = model.score(self.X, self.lengths)

                if max_score is None or max_score < score:
                    max_score = score
                    if final_model is None:
                        final_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                  random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    max_model = final_model

            except:
                pass

        return max_model
