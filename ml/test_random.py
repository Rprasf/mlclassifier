import numpy as np
from sklearn.datasets import make_classification
from ml.train import RandomForestClassifiers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path
#Todo : move ml folder inside api folder before testing.


# Generate a random regression problem
X, y = make_classification(
    n_samples=750, n_features=10, n_informative=8, random_state=1111, n_classes=2, class_sep=2.5, n_redundant=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=1111)

#TODO : to be fixed with file
def test_random_forest():

    model = RandomForestClassifiers()
    model.fit()
    predictions = model.predict(X_test)[:, 1]
    assert roc_auc_score(y_test, predictions) >= 0.95
