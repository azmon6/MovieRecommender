from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import *
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv("smallDataset.csv")

movieTitleEncoder = LabelEncoder()
movieTitleEncoder.fit(data['Series_Title'])
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

data = prepareDatasets(data, tfidf_vectorizer, True, movieTitleEncoder)

# Split data into features and target variable
X = data.drop(columns=["Series_Title"])  # Features
y = data["Series_Title"]  # Target variable

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
X_train = X
y_train = y
rf_model = RandomForestClassifier(max_depth=None, random_state=24, n_estimators=200)
# rf_model.fit(X_train, y_train)

test = [0, 0,
        "The Mayan kingdom is at the height of its opulence and power but the foundations of the empire are beginning to crumble. The leaders believe they must build more temples and sacrifice more people or their crops and citizens will die. Jaguar Paw (Rudy Youngblood), a peaceful hunter in a remote tribe, is captured along with his entire village in a raid. He is scheduled for a ritual sacrifice until he makes a daring escape and tries to make it back to his pregnant wife and son. Action, Adventure, Drama"]
fuckit = text_preprocess(pd.Series(test[2]), tfidf_vectorizer.get_feature_names_out())
fuckit = tfidf_vectorizer.transform(fuckit)

newDF = pd.DataFrame([[test[0], test[1]]], columns=["IMDB_Rating", "Released_Year"])
newVectorDF = pd.DataFrame(fuckit.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
newDF = pd.concat([newDF, newVectorDF], axis=1)
print(newDF)
# print(movieTitleEncoder.inverse_transform(rf_model.predict(newDF)))

Testdata = pd.read_csv("testDataset.csv")

Testdata = prepareDatasets(Testdata, tfidf_vectorizer, False, movieTitleEncoder)
Testdata = Testdata.drop(columns='Actors')

Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
Test_Target = Testdata["Series_Title"]  # Target variable
# print(rf_model.score(Test_Features, Test_Target))

CV_Features = pd.concat([X,Test_Features])
CV_Target = pd.concat([y,Test_Target])

from sklearn.model_selection import GridSearchCV

# parameters = {'criterion': ("gini", "entropy"), 'max_depth': [None]}
#
parameters = {'criterion': ("gini", "entropy"), 'max_depth': [None,20, 200, 800],
              'max_features': ["sqrt", "log2", 0.3, 0.8, None, 4, 5],'bootstrap': [False, True], 'n_jobs': [-1],
              'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}
# parameters = {'criterion': ("gini", "log_loss"), 'max_depth': [800],
#               'max_features': ["log2", None], 'n_jobs': [-1],
#               'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}

# parameters = {'criterion': ("gini", "entropy", "log_loss"), 'max_depth': [None, 100, 200, 300, 400, 800, 1000],
#               'max_features': ["sqrt", "log2", 0.3, 0.1, 0.8, "None"], 'bootstrap': [False, True], 'n_jobs': [-1],
#               'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}
clf = GridSearchCV(rf_model, parameters, scoring="accuracy", verbose=20, cv=2, n_jobs=-1)

clf.fit(CV_Features, CV_Target)

pickle.dump(clf, open('finalized_model1.sav','wb'))