
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data= pd.read_csv("/content/healthcare-dataset-stroke-data.csv")

data.head(10)

data = data.dropna()
data = data.drop(['id'], axis=1)
data = data.drop(['work_type'], axis=1)
data

data = data.replace("Yes", 1)
data = data.replace("No", 0)
data = data.replace("Urban", 1)
data = data.replace("Rural", 0)
data = data.replace(["Male", "Female", "Other"], [1, 2, 0])
data = data.replace(["formerly smoked","never smoked","smokes","Unknown"], [1,0,2,3])
data

data["stroke"].value_counts()

datax = data.drop("stroke", axis=1)
datay = data["stroke"]

datax.head()
datay

train_X, test_X, train_y, test_y = train_test_split(datax, datay, test_size=0.2, random_state = 42)

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_X)

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Create a range of values for n_neighbors
n_neighbors_range = range(1, 15)

# Perform cross-validation for each value of n_neighbors
cv_scores = []
for n_neighbors in n_neighbors_range:
    knn.n_neighbors = n_neighbors
    scores = cross_val_score(knn, X_scaled, train_y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Create a heatmap
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), cv_scores, marker='o')
plt.title('Cross-Validation Scores for KNN')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, 15))
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import time

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)

# Decision Tree Classifier
start_time = time.time()
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, train_y)
end_time = time.time()

dt_training_time = end_time - start_time

start_time = time.time()
dt_pred = dt_classifier.predict(X_test_scaled)
end_time = time.time()

dt_testing_time = end_time - start_time

# Support Vector Machine (SVM)
start_time = time.time()
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(X_train_scaled, train_y)
end_time = time.time()

svm_training_time = end_time - start_time

start_time = time.time()
svm_pred = svm_classifier.predict(X_test_scaled)
end_time = time.time()

svm_testing_time = end_time - start_time

#knn classifier
start_time = time.time()
knn_classifier = KNeighborsClassifier(n_neighbors=11)
knn_classifier.fit(X_train_scaled, train_y)
end_time = time.time()

knn_training_time = end_time - start_time

start_time = time.time()
knn_pred = knn_classifier.predict(X_test_scaled)
end_time = time.time()

knn_testing_time = end_time - start_time

# Evaluation metrics
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, roc_auc, conf_matrix

# Decision Tree evaluation
dt_acc, dt_prec, dt_rec, dt_f1, dt_roc_auc, dt_conf_matrix = evaluate_model(test_y, dt_pred)

# SVM evaluation
svm_acc, svm_prec, svm_rec, svm_f1, svm_roc_auc, svm_conf_matrix = evaluate_model(test_y, svm_pred)

# KNN evaluation
knn_acc, knn_prec, knn_rec, knn_f1, knn_roc_auc, knn_conf_matrix = evaluate_model(test_y, knn_pred)

print("Decision Tree Classifier:")
print("Accuracy:", dt_acc)
print("Precision:", dt_prec)
print("Recall:", dt_rec)
print("F1 Score:", dt_f1)
print("ROC AUC Score:", dt_roc_auc)
print("Training Time:", dt_training_time)
print("Testing Time:", dt_testing_time)

plt.figure(figsize=(4, 4))
print(dt_conf_matrix)
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision tree classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nSupport Vector Machine (SVM):")
print("Accuracy:", svm_acc)
print("Precision:", svm_prec)
print("Recall:", svm_rec)
print("F1 Score:", svm_f1)
print("ROC AUC Score:", svm_roc_auc)
print("Training Time:", svm_training_time)
print("Testing Time:", svm_testing_time)

plt.figure(figsize=(4, 4))
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - SVM classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nK-nearest Neighbours (KNN):")
print("Accuracy:", knn_acc)
print("Precision:", knn_prec)
print("Recall:", knn_rec)
print("F1 Score:", knn_f1)
print("ROC AUC Score:", knn_roc_auc)
print("Training Time:", knn_training_time)
print("Testing Time:", knn_testing_time)

plt.figure(figsize=(4, 4))
sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - KNN classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot ROC Curves
plt.figure(figsize=(10, 6))
RocCurveDisplay.from_estimator(dt_classifier, X_test_scaled,test_y)
# RocCurveDisplay.from_estimator(svm_classifier, X_test_scaled, test_y)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
#RocCurveDisplay.from_estimator(dt_classifier, X_test_scaled,test_y)
RocCurveDisplay.from_estimator(svm_classifier, X_test_scaled, test_y)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
#RocCurveDisplay.from_estimator(dt_classifier, X_test_scaled,test_y)
RocCurveDisplay.from_estimator(knn_classifier, X_test_scaled, test_y)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()

