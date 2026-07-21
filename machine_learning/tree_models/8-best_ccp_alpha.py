#!/usr/bin/env python3
""" module that selects the best ccp_alpha among pruned trees """


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """ Returns the best alpha and its classifier """

    # highest test accuracy
    max_test = max(test_scores)
    candidates = [i for i, s in enumerate(test_scores) if s == max_test]

    # smallest train-test gap
    if len(candidates) > 1:
        gaps = [abs(train_scores[i] - test_scores[i]) for i in candidates]
        min_gap = min(gaps)
        candidates = [i for i, g in zip(candidates, gaps) if g == min_gap]

    # largest ccp_alpha
    best_idx = candidates[0]
    for i in candidates:
        if ccp_alphas[i] > ccp_alphas[best_idx]:
            best_idx = i

    return ccp_alphas[best_idx], clfs[best_idx]
