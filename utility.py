import warnings

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)


def text_preprocess(ds: pd.Series, vocabulary=[]) -> pd.Series:
    for m in range(len(ds)):
        main_words = re.sub('[^a-zA-Z]', ' ', ds[m])  # Retain only alphabets
        main_words = (main_words.lower()).split()
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]  # Remove stopwords
        if len(vocabulary) != 0:
            main_words = [w for w in main_words if w in vocabulary]
        lem = WordNetLemmatizer()
        main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1]  # Group different forms of the same word

        main_words = ' '.join(main_words)
        ds[m] = main_words
    return ds


def prepareDatasets(ds: pd.DataFrame, tfidf_vectorizer, isFirst, movieTitleEncoder) -> pd.DataFrame:
    ds['Series_Title'] = movieTitleEncoder.transform(ds['Series_Title'])
    ds['IMDB_Rating'] /= 10
    ds['Released_Year'] /= 2000
    ds['Overview'] = ds['Overview'] + ' ' + ds['Genre']
    ds = ds.drop(columns='Genre')
    ds['Overview'] = text_preprocess(ds['Overview'])
    if isFirst:
        X_tfidf = tfidf_vectorizer.fit_transform(
            ds['Overview'])  # Using TF-IDF (Term Frequency-Inverse Document Frequency)
    else:
        X_tfidf = tfidf_vectorizer.transform(ds['Overview'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(),
                            columns=tfidf_vectorizer.get_feature_names_out())  # Convert to DataFrame for visualization
    ds = pd.concat([ds, tfidf_df], axis=1)
    ds = ds.drop(columns=["Overview"])
    ds = ds.sort_index(axis=1)
    return ds
