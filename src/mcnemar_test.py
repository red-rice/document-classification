import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# load predictions
y_true = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy")
y_pred_multi = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_pred_multi.npy")
y_pred_text = np.load("eval_outputs_text_50k/bert_margin_star_50k_y_pred_text.npy")

assert len(y_true) == len(y_pred_text) == len(y_pred_multi)

# correctness
text_correct = (y_pred_text == y_true)
multi_correct = (y_pred_multi == y_true)

# 2x2 table
n11 = np.sum((text_correct == 1) & (multi_correct == 1))
n00 = np.sum((text_correct == 0) & (multi_correct == 0))
n10 = np.sum((text_correct == 1) & (multi_correct == 0))
n01 = np.sum((text_correct == 0) & (multi_correct == 1))

print("\nContingency Table:")
print(f"n11 (both correct) = {n11}")
print(f"n00 (both wrong)   = {n00}")
print(f"n10 (text correct, multi wrong) = {n10}")
print(f"n01 (text wrong, multi correct) = {n01}")

# McNemar test
table = [[n11, n10],
         [n01, n00]]

result = mcnemar(table, exact=False, correction=True)

print("\nMcNemar Test Result:")
print(f"chi2 statistic = {result.statistic}")
print(f"p-value = {result.pvalue}")

# interpretation
if result.pvalue < 0.05:
    print("\nResult: Statistically significant (p < 0.05)")
else:
    print("\nResult: NOT statistically significant")