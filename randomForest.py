from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import *
from ParameterSearch import searchParameters
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Pandas neshto mrunka
pd.options.mode.chained_assignment = None  # default='warn'

def recommendMovie(userPrompt):
    # userPrompt = "Brad Pitt action 2004 casino"
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

    results =singleTest(rf_model,tfidf_vectorizer,movieTitleEncoder,userPrompt)

    recommenderResults = AllData[AllData['Series_Title'].isin(results)]
    recommenderResults = recommenderResults['Series_Title'] + "-" + recommenderResults['Overview']
    print(recommenderResults.to_csv(header=False,index=False,index_label=False))

    # looks cool
    # print(recommenderResults.to_string(header=False,index=False,index_names=False,col_space=0))

    Testdata = pd.read_csv("testDataset.csv")
    Testdata = prepareDatasets(Testdata, tfidf_vectorizer, False, movieTitleEncoder)
    testModel(rf_model,Testdata)

    # SVC_model = make_pipeline(StandardScaler(),SVC(gamma="scale", kernel="linear", random_state=42,probability=True))
    # SVC_model.fit(X,y)
    # singleTest(SVC_model, tfidf_vectorizer, movieTitleEncoder)
    # TestdataSVC = pd.read_csv("testDataset.csv")
    # testModel(SVC_model,tfidf_vectorizer,movieTitleEncoder,TestdataSVC)

    #Neshto ne raboti kakto trqbva
    # kernel = 1.0 * RBF(1.0)
    # GPC = GaussianProcessClassifier(kernel=kernel,random_state=42, max_iter_predict=10, n_jobs=2)
    # GPC.fit(X,y)
    # singleTest(GPC,tfidf_vectorizer,movieTitleEncoder)
    # TestdataGPC = pd.read_csv("testDataset.csv")
    # testModel(GPC,tfidf_vectorizer,movieTitleEncoder,TestdataGPC)


    # Tuk nadolu e da pusnesh da testvash koi parametri na random forest davat nay dobro accuracy
    Test_Features = Testdata.drop(columns=["Series_Title"])  # Features
    Test_Target = Testdata["Series_Title"]  # Target variable
    CV_Features = pd.concat([X, Test_Features])
    CV_Target = pd.concat([y, Test_Target])
    parameters = {'criterion': ("gini", "log_loss"), 'max_depth': [200,800],
                  'max_features': ["log2", None], 'n_jobs': [4],
                  'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}

    # searchParameters(rf_model,CV_Features,CV_Target,parameters)