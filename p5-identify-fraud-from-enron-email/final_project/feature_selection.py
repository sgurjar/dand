from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def positive_features(feature_names, features):
    flags = features.min(axis=0) >= 0
    names = [c for c,f in zip(feature_names, flags) if f]
    f = features[:, flags]
    return names, f

def kbest(feature_names, features, labels):
    # input X must be non-negative
    # only get features with positive values
    feature_names, features = positive_features(feature_names, features)
    score, features = _kbest(features, labels)
    return sorted( zip(feature_names, score), key=lambda x:x[1], reverse=True )

def _kbest(X, Y):
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, Y)
    features = fit.transform(X)  # Input X must be non-negative
    return (fit.scores_, features)

def rfe(X, Y, n_features_to_select = None):
    """n_features_to_select: The number of features to select. If None,
            half of the features are selected.
    """
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select)
    fit = rfe.fit(X, Y)
    return fit.n_features_, fit.support_, fit.ranking_

def pca(X, Y, n_components=10):
    pca = PCA(n_components=n_components)
    fit = pca.fit(X)
    #set_printoptions(precision=3)
    #print("Explained Variance: %s") % fit.explained_variance_ratio_
    #print(fit.components_)
    # explained_variance_ratio_ how imp the feature is ?
    # higher is better
    return fit.explained_variance_ratio_, fit.components_

def extra_trees(X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    # larger the score, the more important the feature.
    # print(model.feature_importances_)
    return model.feature_importances_