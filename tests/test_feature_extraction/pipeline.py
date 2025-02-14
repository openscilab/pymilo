from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymilo.utils.test_pymilo import pymilo_classification_test


MODEL_NAME = "Pipeline"


def pipeline():
    corpus = ['this is the first document',
            'this document is the second document',
            'and this is the third one',
            'is this the first document']

    labels = ['A', 'B', 'A', 'B']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('count', CountVectorizer(vocabulary=['this', 'document', 'first', 'is', 'second', 'the', 'and', 'one'])),
        ('tfid', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    pipe.fit(X_train, y_train)
    pymilo_classification_test(pipe, MODEL_NAME, (X_test, y_test))
