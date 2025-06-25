# lib-ml

**Overview**
A shared library for preprocessing restaurant review data. Used by both `model-training` and `model-service` to ensure consistency. The personal contribution can be seen from `ACTIVITY.md`.  

#### **Features**

- Standardizes text preprocessing (e.g., tokenization, normalization).
- Versioned releases via GitHub Packages.

#### **Installation**

Install the library via GitHub Packages:

```
pip install git+https://github.com/remla2025-team16/libml@v0.0.8
```

#### **Local Build & Install**

To build and install the library locally for development or testing:

1. Clone the repository and navigate to the `lib-ml` directory.
2. Build the package:
    ```
    python -m build
    ```
3. Install the generated wheel file:
    ```
    pip install dist/lib_ml-*.whl
    ```
Now you can import and use `libml` in your local Python environment.
4. To remove the library from your local Python environment, run:
    ```
    pip uninstall lib-ml
    ```
#### **Usage**
1. Use the defined pipeline by:
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
2. Use data preprocessing by:
    ```python
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, 'data', 'a1_RestaurantReviews_HistoricDump.tsv')
    vectorizer_output = os.path.join(project_root, 'output', 'c1_BoW_Sentiment_Model.pkl')
    test_size = 0.2                                  
    random_state = 42                                
    vectorizer_input = None                          

    # Run preprocessing
    result = data_preprocessing(
        filepath=filepath,
        test_size=test_size,
        random_state=random_state,
        vectorizer_output=vectorizer_output,
        vectorizer_input=vectorizer_input
    )

    # Print results summary
    X_train, X_test, y_train, y_test = result
    ```

