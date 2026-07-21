#!/usr/bin/env python3
""" module that selects the best ccp_alpha among pruned trees """


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """ Returns the best alpha and its classifier """

    best_score = max(test_scores)
    best_index = None

    for i in range(len(test_scores)):
        if test_scores[i] != best_score:
            continue
        if best_index is None:
            best_index = i
            continue

        gap = train_scores[i] - test_scores[i]
        best_gap = train_scores[best_index] - test_scores[best_index]

        if gap < best_gap:
            best_index = i
        elif gap == best_gap and ccp_alphas[i] > ccp_alphas[best_index]:
            best_index = i

    return ccp_alphas[best_index], clfs[best_index]
