import warnings

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

#Podavash mu string kolona ot DataFrame i ti vrushta nay vajnite dumi za da go vectorizirash
#Purviq put go puskash s prazen vocabulary
#Sledvashtite puti go vikash s vocabulary = tfidf_vectorizer.get_feature_names_out()
#Zashtoto tfidf_vectorizer.get_feature_names_out() sa ti featurite na modela
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

#Tova prigotvq dataset-a da moje da se puska kum modela za fit/test
#Strukturata na dataset-a e hardcode-nata zasega
def prepareDatasets(ds: pd.DataFrame, tfidf_vectorizer, isFirst, movieTitleEncoder) -> pd.DataFrame:
    ds['Series_Title'] = movieTitleEncoder.transform(ds['Series_Title']) #Imenata gi pravim na chisla
    ds['IMDB_Rating'] /= 10 #Normalizirane na dannite
    ds['Released_Year'] /= 2000 #Normalizirane na dannite
    ds['Overview'] = ds['Overview'] + ' ' + ds['Genre'] #Vsichko koeto e string go butame v edna kolona
    ds = ds.drop(columns='Genre') #I gi premahvame posle
    ds['Overview'] = text_preprocess(ds['Overview']) #Vzimame samo nay vajnite dumi
    if isFirst: #Purviq put kazvame na vectorizer-a s koi dumi rabotim
        X_tfidf = tfidf_vectorizer.fit_transform(
            ds['Overview'])  # Using TF-IDF (Term Frequency-Inverse Document Frequency)
    else: #Sledvashtite puti ignorirame dumi koito ne sme vijdali
        X_tfidf = tfidf_vectorizer.transform(ds['Overview'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(),
                            columns=tfidf_vectorizer.get_feature_names_out())  # Convert to DataFrame for visualization
    ds = pd.concat([ds, tfidf_df], axis=1) #Dobavqme vektoriziranite dumi v dataset-a
    ds = ds.drop(columns=["Overview"]) # I mahame string kolonite zashtoto modela raboti samo s chisla
    ds = ds.sort_index(axis=1) #Sortirash gi v pravilniq red inache modela ne raboti
    return ds
