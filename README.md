# lib-ml

**Overview**
A shared library for preprocessing restaurant review data. Used by both `model-training` and `model-service` to ensure consistency. The personal contribution can be seen from `ACTIVITY.md`.  

#### **Features**

- Standardizes text preprocessing (e.g., tokenization, normalization).
- Versioned releases via GitHub Packages/PyPI.

#### **Installation**

Install the library via PyPI:

```
# For Python (using GitHub Packages)
pip install git+https://github.com/remla2025-team16/libml@v1.0.0
```

#### **Usage**

```python
from libml.preprocessing import build_pipeline

# Build a text classification pipeline
pipeline = build_pipeline()

# Example usage with training data
X_train = ["This restaurant was amazing!", "The food was terrible."]
y_train = [1, 0]  # Labels: 1 = positive, 0 = negative

pipeline.fit(X_train, y_train)

# Predict sentiment for new reviews
predictions = pipeline.predict(["The service was excellent!"])
print(predictions)  # Output: [1]
```

