from sklearn import ensemble
0.75091
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, verbose=2)
}