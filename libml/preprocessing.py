from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_vectorizer():
    
    return CountVectorizer(
        lowercase=True,
        stop_words='english',   
        max_features=5000       
    )

def build_pipeline():
    
    vectorizer = build_vectorizer()
    classifier = LogisticRegression(max_iter=200)

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    return pipeline
