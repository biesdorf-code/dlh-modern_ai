#!/usr/bin/env python3
""" module that selects the best ccp_alpha among pruned trees """


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """ Returns the best alpha and its classifier """

    best_score = max(test_scores)
    candidates = [i for i, score in enumerate(test_scores)
                  if score == best_score]

    best_index = min(candidates,
                     key=lambda i: (train_scores[i] - test_scores[i],
                                    -ccp_alphas[i]))

    return ccp_alphas[best_index], clfs[best_index]
