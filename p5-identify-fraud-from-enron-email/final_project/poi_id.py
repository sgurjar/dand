#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

# ignore sklearn DeprecationWarning
def ignore_sklearn_DeprecationWarning():
    def warn(*args, **kwargs): pass
    import warnings
    warnings.warn = warn
ignore_sklearn_DeprecationWarning()

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# describe dataset
def describe_dataset():
    print('### dataset size=%d' % len(data_dict))
    feature_names = set(k2 for k, v in data_dict.items()
                            for k2 in v.keys() if k2 != 'poi') # poi is label
    print '### features: size=%d, names=[%s] ' % ( len(feature_names),
                                            ', '.join(feature_names) )

    import pandas as pd

    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame.from_dict(data_dict, orient='index')\
            .reset_index().rename(columns={'index':'employee_name'})\
            .convert_objects(convert_numeric=True)
    df.info()
    print(df.describe().transpose())
    print(df.head(2).transpose())
    # POI's
    print(df[df.poi][['employee_name','poi']].shape)

    # sample record
    k = data_dict.keys()[42]
    print {k: data_dict[k]}

def visualize_dataset():
    import pandas as pd
    import matplotlib.pyplot as pyplot
    from pandas.tools.plotting import scatter_matrix

    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame.from_dict(data_dict, orient='index')\
            .reset_index().rename(columns={'index':'employee_name'})\
            .convert_objects(convert_numeric=True)
    #df.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False)
    #df.hist()
    scatter_matrix(df)
    pyplot.show()

### Task 2: Remove outliers

def find_outliers():
    import numpy as np

    def top_k_feature_values(_data_dict, feature_name, k=10):
        values = [(key, val[feature_name]) for key, val in _data_dict.items()
                        if val[feature_name] != 'NaN']
        values.sort(key=lambda x:x[1], reverse=True)
        median = np.median(map(lambda x: x[1], filter(lambda x: x[1]!='NaN', values)))
        top_k = values[:k]
        top_k = [(val[0], val[1], round((val[1]/median)*100,2)) for val in top_k]
        return top_k

    def printit(msg, itr):
        print "\n###",msg
        for x in itr:
            print x

    printit('salary', top_k_feature_values(data_dict, 'salary', 10))
    printit('bonus', top_k_feature_values(data_dict, 'bonus', 10))
    printit('total_payments', top_k_feature_values(data_dict, 'total_payments', 10))
    printit('total_stock_value', top_k_feature_values(data_dict, 'total_stock_value', 10))

    # key TOTAL's salary is ~10000% above median salary, bonus is ~12000%
    #  above median bonus, so this is definately outlier, also TOTAL doesnt
    #  sound like a name rather TOTAL of all rows.
    #
    # There are 3 more employees whos salary is above 400% of median
    # but that expected and this information is useful to find POI
    ### salary
    # ('TOTAL', 26704229, 10271.02)
    # ('SKILLING JEFFREY K', 1111258, 427.41)
    # ('LAY KENNETH L', 1072321, 412.44)
    # ('FREVERT MARK A', 1060932, 408.06)
    # ('PICKERING MARK R', 655037, 251.94)
    # ('WHALLEY LAWRENCE G', 510364, 196.3)
    # ('DERRICK JR. JAMES V', 492375, 189.38)
    # ('FASTOW ANDREW S', 440698, 169.5)
    # ('SHERRIFF JOHN R', 428780, 164.92)
    # ('RICE KENNETH D', 420636, 161.79)
    ###########################################
    # Lavorato and Lay's bonus is over 900% of median bonus
    ### bonus
    # ('TOTAL', 97343619, 12652.3)
    # ('LAVORATO JOHN J', 8000000, 1039.81)
    # ('LAY KENNETH L', 7000000, 909.83)
    # ('SKILLING JEFFREY K', 5600000, 727.86)
    # ('BELDEN TIMOTHY N', 5249999, 682.37)
    # ('ALLEN PHILLIP K', 4175000, 542.65)
    # ('KITCHEN LOUISE', 3100000, 402.92)
    # ('WHALLEY LAWRENCE G', 3000000, 389.93)
    # ('DELAINEY DAVID W', 3000000, 389.93)
    # ('MCMAHON JEFFREY', 2600000, 337.94)
    ###########################################
    # Ken Lay's total_payments is above 9000% of median
    ### total_payments
    # ('TOTAL', 309886585, 28135.88)
    # ('LAY KENNETH L', 103559793, 9402.62)
    # ('FREVERT MARK A', 17252530, 1566.43)
    # ('BHATNAGAR SANJAY', 15456290, 1403.34)
    # ('LAVORATO JOHN J', 10425757, 946.6)
    # ('SKILLING JEFFREY K', 8682716, 788.34)
    # ('MARTIN AMANDA K', 8407016, 763.31)
    # ('BAXTER JOHN C', 5634343, 511.57)
    # ('BELDEN TIMOTHY N', 5501630, 499.52)
    # ('DELAINEY DAVID W', 4747979, 431.09)
    ###########################################

# uncomment to see how outlier was found
# find_outliers()

# remove outlier
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

financial_features=['deferral_payments', 'total_payments',
    'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
    'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
    'long_term_incentive', 'restricted_stock', 'director_fees']

email_features=['to_messages', 'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list += financial_features + email_features

def create_new_feature(_data_dict, _features_list):
    """create new feature fraction of emails sent to POI, fraction_to_poi"""
    def fraction(part, whole):
        if part != 'NaN' and whole != 'NaN' and whole != 0:
            return part/float(whole)
        else:
            return 0
    for name, data in _data_dict.items():
        # this person sends email to one of the POI
        data['fraction_to_poi'] = fraction(data['from_this_person_to_poi'],
                                           data['from_messages'])

        # there are general emails from top executives- announcement, memos etc
        # thats why this features is kind of polluted and doesnt necessarily
        # represent communication between specific POI and recvr of the email
        data['fraction_from_poi'] = fraction(data['from_poi_to_this_person'],
                                              data['to_messages'])

    _features_list += ['fraction_to_poi','fraction_from_poi']

# new features reduces recall and precision of KNN and DTC
# create_new_feature(data_dict, features_list)
my_feature_list = features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

def compare_algos(_features_train, _labels_train, _dataset, _features_list):
    """compare various classification algo"""

    print '###features_list:', _features_list
    print '###features_train: size=', len(_features_train), \
                'no_of_features:', len(_features_train[0])

    clfs = [
            ('RandomForestClassifier'    , RandomForestClassifier(random_state=42)),
            ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),
            ('LogisticRegression'        , LogisticRegression(random_state=42)),
            ('GaussianNB'                , GaussianNB()),
            ('SVM'                       , SVC(random_state=42)),
            ('KNeighborsClassifier'      , KNeighborsClassifier()),
            ('DecisionTreeClassifier'    , DecisionTreeClassifier(random_state=42)),
            ('AdaBoostClassifier'        , AdaBoostClassifier(random_state=42))
        ]

    for name, clf in clfs:
        pipeline = Pipeline(steps=[
            #('scaler', StandardScaler()),
            ('pca', PCA(n_components=5)),
            (name, clf)])
        print "\n\n###", name
        pipeline.fit(_features_train, _labels_train)
        test_classifier(pipeline, _dataset, _features_list)

    """
    ###features_list: ['poi', 'salary', 'deferral_payments', 'total_payments',
        'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
        'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
        'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages',
        'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
        'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']
    ###features_train: size= 100 no_of_features: 21


    ### RandomForestClassifier
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('RandomForestClassifier', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes...stimators=10, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False))])
        Accuracy: 0.85713	Precision: 0.40654	Recall: 0.15550	F1: 0.22495	F2: 0.17741
        Total predictions: 15000	True positives:  311	False positives:  454	False negatives: 1689	True negatives: 12546

    ### GradientBoostingClassifier
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('GradientBoostingClassifier', GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
            ...rs=100, presort='auto', random_state=42,
                  subsample=1.0, verbose=0, warm_start=False))])
        Accuracy: 0.84320	Precision: 0.36944	Recall: 0.24900	F1: 0.29749	F2: 0.26637
        Total predictions: 15000	True positives:  498	False positives:  850	False negatives: 1502	True negatives: 12150

    ### LogisticRegression
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('LogisticRegression', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=42, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False))])
        Accuracy: 0.83680	Precision: 0.31240	Recall: 0.18650	F1: 0.23356	F2: 0.20285
        Total predictions: 15000	True positives:  373	False positives:  821	False negatives: 1627	True negatives: 12179

    ### GaussianNB
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('GaussianNB', GaussianNB(priors=None))])
        Accuracy: 0.86393	Precision: 0.47931	Recall: 0.23750	F1: 0.31762	F2: 0.26415
        Total predictions: 15000	True positives:  475	False positives:  516	False negatives: 1525	True negatives: 12484

    ### SVM
    Got a divide by zero when trying out: Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('SVM', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False))])
    Precision or recall may be undefined due to a lack of true positive predicitons.

    ### KNeighborsClassifier
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('KNeighborsClassifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform'))])
        Accuracy: 0.88767	Precision: 0.74119	Recall: 0.24200	F1: 0.36487	F2: 0.27967
        Total predictions: 15000	True positives:  484	False positives:  169	False negatives: 1516	True negatives: 12831

    ### DecisionTreeClassifier
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('DecisionTreeClassifier', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=42, splitter='best'))])
        Accuracy: 0.81340	Precision: 0.29628	Recall: 0.29050	F1: 0.29336	F2: 0.29164
        Total predictions: 15000	True positives:  581	False positives: 1380	False negatives: 1419	True negatives: 11620

    ### AdaBoostClassifier
    Pipeline(steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('AdaBoostClassifier', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=42))])
        Accuracy: 0.81527	Precision: 0.24919	Recall: 0.19150	F1: 0.21657	F2: 0.20080
        Total predictions: 15000	True positives:  383	False positives: 1154	False negatives: 1617	True negatives: 11846
    """
    # DecisionTreeClassifier is close to .3 Precision and Recall
    # KNeighborsClassifier has very high Precision and Recall close to .3
    # we will try to tune these 2 algos.

def tuneKNN(_features_train, _labels_train, _dataset, _features_list):
    print '###_features_list %s %s' % (len(_features_list), _features_list)
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca'   , PCA(random_state=42)),
        ('knn'   , KNeighborsClassifier())])

    n_components = [1, 2, 3, 5, 7, 9, 11, 12, 14]
    n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = [1,2,3,4]

    estimator = GridSearchCV(pipeline, dict(
                    pca__n_components=n_components,
                    knn__n_neighbors=n_neighbors,
                    knn__p=p))
    estimator.fit(_features_train, _labels_train)
    print("Best: %f using %s" % (estimator.best_score_, estimator.best_params_))
    clf = estimator.best_estimator_
    clf.fit(_features_train, _labels_train)
    test_classifier(clf, _dataset, _features_list)
    return clf

def tuneDTC(_features_train, _labels_train, _dataset, _features_list):
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca'   , PCA(random_state=42)),
        ('dtc'   , DecisionTreeClassifier(random_state=42))])

    estimator = GridSearchCV(pipeline, dict(
                            dtc__criterion = ['gini','entropy'],
                            pca__n_components = [1, 2, 3, 5, 7, 9, 11, 12, 14]
                            ))
    estimator.fit(_features_train, _labels_train)
    print("Best: %f using %s" % (estimator.best_score_, estimator.best_params_))
    clf = estimator.best_estimator_
    clf.fit(_features_train, _labels_train)
    test_classifier(clf, _dataset, _features_list)
    return clf

### uncomment to see dataset summarization
# describe_dataset()

### uncomment to see dataset visualization
# visualize_dataset()

### uncomment to compare algorithms
# compare_algos(features_train, labels_train, my_dataset, features_list)

### uncomment to see KNN
# tuneKNN(features_train, labels_train, my_dataset, features_list)

# DecisionTreeClassifier
clf = tuneDTC(features_train, labels_train, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)