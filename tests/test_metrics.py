from src.experiment import _scores


def test_scores_returns_specificity_and_fp_rate():
    # TN=3, FP=1, FN=1, TP=2  -> specificity = 3/4 = 0.75, fp_rate = 0.25
    truths = [0, 0, 0, 0, 1, 1, 1]
    preds  = [0, 0, 0, 1, 0, 1, 1]
    acc, prec, rec, spec, fpr = _scores(truths, preds)
    assert abs(spec - 0.75) < 1e-9
    assert abs(fpr - 0.25) < 1e-9


def test_scores_specificity_no_negatives_is_zero_safe():
    # all-positive truths: no true negatives -> specificity defined as 0.0, no crash
    truths = [1, 1, 1]
    preds  = [1, 0, 1]
    acc, prec, rec, spec, fpr = _scores(truths, preds)
    assert spec == 0.0 and fpr == 0.0
