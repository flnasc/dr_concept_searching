from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index= np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)
num1 = 0
for num in y_train:
	if num == 5:
		num1 += 1
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
some_digit = X[36000]
print(num1)

# #forest
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_scores_forest = y_probas_forest[:,1]
# fpr_forest, tpr_forest, thresholds_forest= roc_curve(y_train_5, y_scores_forest)

# #sgd
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# y_scores =  cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method = "decision_function")
# precisions, recalls,thresholds = precision_recall_curve(y_train_5, y_scores)
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)



# def pre_rec_plot(precisions, recalls, thresholds):
# 	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# 	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# 	plt.xlabel("Threshold")
# 	plt.legend(loc="upper left")
# 	plt.ylim([0,1])

# def roc_curve_plot(fpr, tpr, label=None):
# 	plt.plot(fpr, tpr, linewidth=2, label=label)
# 	plt.plot([0,1], [0,1], 'k--')
# 	plt.axis([0,1,0,1])
# 	plt.xlabel('False Positive Rate')
# 	plt.ylabel('True Positive Rate')

# plt.plot(fpr,tpr,"b:", label="SGD")
# roc_curve_plot(fpr_forest, tpr_forest, "Random Forest")
# plt.show()




