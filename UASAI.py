# UAS Machine Learning
# Decision Tree - Dataset Iris
# Nama  : Septiano Alvian Ismau
# NIM   : 231011400813
# Kelas : 05TPLE013

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1. Load Dataset Iris
# ================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("=== Data Awal ===")
print(df.head())

# ================================
# 2. Preprocessing Data
# ================================
X = df.drop('target', axis=1)
y = df['target']

# ================================
# 3. Split Data Training & Testing
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 4. Build Decision Tree Model
# ================================
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 5. Evaluasi Model
# ================================
y_pred = model.predict(X_test)

print("\n=== Evaluasi Model ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ================================
# 6. Visualisasi Decision Tree
# ================================
plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree - Dataset Iris")
plt.show()
