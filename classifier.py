"""
    classifier algo     
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


# Define the classifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
