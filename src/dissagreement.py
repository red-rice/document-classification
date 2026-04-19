import numpy as np
import matplotlib.pyplot as plt

# UPDATE PATHS (VERY IMPORTANT)
y_true = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy")
y_pred_multi = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_pred_multi.npy")
y_pred_text = np.load("eval_outputs_text_50k/bert_margin_star_50k_y_pred_text.npy")

# correctness
text_correct = (y_pred_text == y_true)
multi_correct = (y_pred_multi == y_true)

# counts
n11 = np.sum((text_correct == 1) & (multi_correct == 1))
n00 = np.sum((text_correct == 0) & (multi_correct == 0))
n10 = np.sum((text_correct == 1) & (multi_correct == 0))
n01 = np.sum((text_correct == 0) & (multi_correct == 1))

# plot
labels = ["Both Correct (n11)", "Both Wrong (n00)", "Text Only Correct (n10)", "Multimodal Only Correct (n01)"]
values = [n11, n00, n10, n01]

plt.figure()
plt.bar(labels, values)
plt.title("Model Agreement vs Disagreement")
plt.xticks(rotation=30)
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Multimodal
cm_multi = confusion_matrix(y_true, y_pred_multi)

plt.figure()
plt.imshow(cm_multi)
plt.title("Confusion Matrix (Multimodal)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

# Text-only
cm_text = confusion_matrix(y_true, y_pred_text)

plt.figure()
plt.imshow(cm_text)
plt.title("Confusion Matrix (Text-only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()