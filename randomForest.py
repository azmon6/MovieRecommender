from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import *
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning) # Pandas neshto mrunka
pd.options.mode.chained_assignment = None  # default='warn'


data = pd.read_csv("smallDataset.csv") # Tova e trenirashtite danni

movieTitleEncoder = LabelEncoder() # Tova pravi Imenata na filmite kum chisla i obratno
movieTitleEncoder.fit(data['Series_Title'])
tfidf_vectorizer = TfidfVectorizer(stop_words='english') 

# Tova pravi vsichki dataset-i da sa s pravilnata struktura za modela
data = prepareDatasets(data, tfidf_vectorizer, True, movieTitleEncoder)

X = data.drop(columns=["Series_Title"])  # Features
y = data["Series_Title"]  # Target variable

rf_model = RandomForestClassifier(max_depth=None, random_state=24, n_estimators=200)
rf_model.fit(X, y)

#Tova e na ruka da pusnesh da vidi s daden input kakvo vrushta i sledva sushtite stupki
#kato pri testvane s dataset
testInput = [0.78, 0.3,
        "Tuk slagash vsicko koeto e string"]
obrabotenText = text_preprocess(pd.Series(testInput[2]), tfidf_vectorizer.get_feature_names_out())
obrabotenText = tfidf_vectorizer.transform(obrabotenText)

#pravish DataFrame koyto da podadesh na predict da ti vurne recommended film
newDF = pd.DataFrame([[testInput[0], testInput[1]]], columns=["IMDB_Rating", "Released_Year"])
newVectorDF = pd.DataFrame(obrabotenText.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
newDF = pd.concat([newDF, newVectorDF], axis=1)
print(newDF)
#rezultata shte e chislo i trqbva da go decode-nesh
print(movieTitleEncoder.inverse_transform(rf_model.predict(newDF)))

#Dannite za test minavat prez sushtite stupki kato dannite za training
Testdata = pd.read_csv("testDataset.csv")

Testdata = prepareDatasets(Testdata, tfidf_vectorizer, False, movieTitleEncoder)
#Danni koito ne polzvash prosto gi dropvash
Testdata = Testdata.drop(columns='Actors')

Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
Test_Target = Testdata["Series_Title"]  # Target variable
print(rf_model.score(Test_Features, Test_Target))

# Tuk nadolu e da pusnesh da testvash koi parametri na random forest davat nay dobro accuracy


# CV_Features = pd.concat([X,Test_Features])
# CV_Target = pd.concat([y,Test_Target])

# from sklearn.model_selection import GridSearchCV

# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [None]}
#
# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [None,20, 200, 800],
#               'max_features': ["sqrt", "log2", 0.3, 0.8, None, 4, 5],'bootstrap': [False, True], 'n_jobs': [-1],
#               'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}
# parameters = {'criterion': ("gini", "log_loss"), 'max_depth': [800],
#               'max_features': ["log2", None], 'n_jobs': [-1],
#               'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}

# parameters = {'criterion': ("gini", "entropy", "log_loss"), 'max_depth': [None, 100, 200, 300, 400, 800, 1000],
#               'max_features': ["sqrt", "log2", 0.3, 0.1, 0.8, "None"], 'bootstrap': [False, True], 'n_jobs': [-1],
#               'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}
# clf = GridSearchCV(rf_model, parameters, scoring="accuracy", verbose=20, cv=2, n_jobs=-1)
# 
# clf.fit(CV_Features, CV_Target)
# 
# pickle.dump(clf, open('finalized_model1.sav','wb'))