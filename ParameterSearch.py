import pickle
from sklearn.model_selection import GridSearchCV


def searchParameters(model, Features, Target, parameters={}):
    if parameters == {}:
        parameters = {'criterion': ("gini", "entropy", "log_loss"), 'max_depth': [None, 100, 200, 300, 400, 800, 1000],
                      'max_features': ["sqrt", "log2", 0.3, 0.1, 0.8, None], 'bootstrap': [False, True], 'n_jobs': [-1],
                      'random_state': [42], 'class_weight': [None, "balanced", "balanced_subsample"]}

    clf = GridSearchCV(model, parameters, scoring="accuracy", verbose=20, cv=2, n_jobs=-1)

    clf.fit(Features, Target)

    pickle.dump(clf, open('finalized_model1.sav', 'wb'))
