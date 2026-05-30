import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from palmerpenguins import load_penguins

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

penguins = load_penguins()

penguins = penguins.dropna()

X_reg = penguins[['flipper_length_mm']]
y_reg = penguins['body_mass_g']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train_r, y_train_r)

y_pred_r = lr.predict(X_test_r)

print("LINEAR REGRESSION")
print("MSE:", mean_squared_error(y_test_r, y_pred_r))
print("R2 Score:", r2_score(y_test_r, y_pred_r))

features = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

X = penguins[features]
y = penguins['species']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test, dt_pred))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].scatter(
    X_test_r['flipper_length_mm'],
    y_test_r,
    alpha=0.6
)

sorted_idx = X_test_r['flipper_length_mm'].argsort()

ax[0].plot(
    X_test_r['flipper_length_mm'].iloc[sorted_idx],
    y_pred_r[sorted_idx],
    color='red'
)

ax[0].set_title('Linear Regression')
ax[0].set_xlabel('Flipper Length')
ax[0].set_ylabel('Body Mass')

cm_dt = confusion_matrix(y_test, dt_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=dt.classes_
).plot(ax=ax[1], cmap='Blues', colorbar=False)

ax[1].set_title('Decision Tree')

cm_rf = confusion_matrix(y_test, rf_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=rf.classes_
).plot(ax=ax[2], cmap='Greens', colorbar=False)

ax[2].set_title('Random Forest')

plt.tight_layout()
plt.show()

y_test_bin = label_binarize(
    y_test,
    classes=rf.classes_
)

rf_prob = rf.predict_proba(X_test)

plt.figure(figsize=(8,6))

for i in range(len(rf.classes_)):
    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        rf_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{rf.classes_[i]} AUC={roc_auc:.2f}"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')

plt.legend()

plt.show()