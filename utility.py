import warnings

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
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
    ds = ds.drop(columns=["IMDB_Rating", "Released_Year"])
    ds = ds.sort_index(axis=1) #Sortirash gi v pravilniq red inache modela ne raboti
    return ds

def singleTest(model,tfidf_vectorizer,movieTitleEncoder):
    # Tova e na ruka da pusnesh da vidi s daden input kakvo vrushta i sledva sushtite stupki
    # kato pri testvane s dataset
    testInput = [0.78, 2006/2000,
                 "Batman pls chovek Joker Batman Batman Batman Batman Joker Joker Catwoman Gotham Gotham "]
    obrabotenText = text_preprocess(pd.Series(testInput[2]), tfidf_vectorizer.get_feature_names_out())
    obrabotenText = tfidf_vectorizer.transform(obrabotenText)

    # pravish DataFrame koyto da podadesh na predict da ti vurne recommended film
    newDF = pd.DataFrame([[testInput[0], testInput[1]]], columns=["IMDB_Rating", "Released_Year"])
    newVectorDF = pd.DataFrame(obrabotenText.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    newDF = newDF.drop(columns=["IMDB_Rating", "Released_Year"])
    newDF = pd.concat([newDF, newVectorDF], axis=1)
    print(newDF)
    # rezultata shte e chislo i trqbva da go decode-nesh
    # print(movieTitleEncoder.inverse_transform(model.predict(newDF)))
    # print(movieTitleEncoder.inverse_transform([pd.Series(model.predict_proba(newDF)[0]).idxmax()]))
    test = np.array(model.predict_proba(newDF)[0])
    print(test.argsort()[-5:])
    print(movieTitleEncoder.inverse_transform(test.argsort()[-5:]))


# # Dannite za test minavat prez sushtite stupki kato dannite za training
# Testdata = pd.read_csv("testDataset.csv")
# Testdata = prepareDatasets(Testdata, tfidf_vectorizer, False, movieTitleEncoder)
# # Danni koito ne polzvash prosto gi dropvash
# Testdata = Testdata.drop(columns='Actors')
#
# Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
# Test_Target = Testdata["Series_Title"]  # Target variable
# print(rf_model.score(Test_Features, Test_Target))
def testModel(model,tfidf_vectorizer,movieTitleEncoder,Testdata):
    # Dannite za test minavat prez sushtite stupki kato dannite za training
    Testdata = prepareDatasets(Testdata, tfidf_vectorizer, False, movieTitleEncoder)
    # Danni koito ne polzvash prosto gi dropvash
    Testdata = Testdata.drop(columns='Actors')

    Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
    Test_Target = Testdata["Series_Title"]  # Target variable
    print(model.score(Test_Features, Test_Target))