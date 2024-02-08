import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample movie descriptions
descriptions = [
    "A thrilling action movie with intense fight scenes.",
    "A touching drama about love and loss.",
    "A hilarious comedy filled with laughter and fun moments for movie."
]

# Using TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(descriptions)

# Convert to DataFrame for visualization
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Vectorizer:")
print(tfidf_df)