from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import *
from ParameterSearch import searchParameters

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Pandas neshto mrunka
pd.options.mode.chained_assignment = None  # default='warn'

AllData = pd.read_csv("smallDataset.csv")
data = pd.read_csv("smallDataset.csv")  # Tova e trenirashtite danni

movieTitleEncoder = LabelEncoder()  # Tova pravi Imenata na filmite kum chisla i obratno
movieTitleEncoder.fit(data['Series_Title'])
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Tova pravi vsichki dataset-i da sa s pravilnata struktura za modela
data = prepareDatasets(data, tfidf_vectorizer, True, movieTitleEncoder)
X = data.drop(columns=["Series_Title"])  # Features
y = data["Series_Title"]  # Target variable
rf_model = RandomForestClassifier(max_depth=None, random_state=24, n_estimators=200, max_features="log2",criterion="gini",class_weight="balanced_subsample",bootstrap=True)
rf_model.fit(X, y)

singleTest(rf_model,tfidf_vectorizer,movieTitleEncoder)
Testdata = pd.read_csv("testDataset.csv")
testModel(rf_model,tfidf_vectorizer,movieTitleEncoder,Testdata)


# Tuk nadolu e da pusnesh da testvash koi parametri na random forest davat nay dobro accuracy
Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
Test_Target = Testdata["Series_Title"]  # Target variable
CV_Features = pd.concat([X, Test_Features])
CV_Target = pd.concat([y, Test_Target])
parameters = {'criterion': ("gini", "log_loss"), 'max_depth': [800],
              'max_features': ["log2", None], 'n_jobs': [-1],
              'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}

# searchParameters(rf_model,CV_Features,CV_Target,parameters)