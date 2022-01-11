from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np


class MlAlgorithms:

    def __init__(self):
        self.rfc = None
        self.svmm = None
        self.knn = None
        self.dtc = None
        self.gnb = None
        self.lr = None
        self.km = None

    def random_forest_classifier(self, training_input, training_output, test_input, test_output, prediction_variables):
        simple_forest_model = RandomForestClassifier(n_estimators=30, criterion='entropy')  # (criterion='entropy')
        simple_forest_model.fit(training_input, training_output)
        prediction_rfc = simple_forest_model.predict(test_input)
        self.rfc = metrics.accuracy_score(prediction_rfc, test_output)

        featimp = pd.Series(simple_forest_model.feature_importances_, index=prediction_variables).\
            sort_values(ascending=False)
        # print(f'\tRandom Forest Classifier variable importance: {round(featimp, 2)}')

        max_features_range = np.arange(1, 6, 1)
        n_estimators_range = np.arange(10, 210, 10)
        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
        rfc_ht = RandomForestClassifier()
        rfc_grid = GridSearchCV(estimator=rfc_ht, param_grid=param_grid, cv=5)
        rfc_grid.fit(training_input, training_output)
        # print(f'\tHyperparameter tuning for RFC -
        # best params: {rfc_grid.best_params_} , best score: {rfc_grid.best_score_}')

    def support_vector_machine(self, training_input, training_output, test_input, test_output):
        svm_model = svm.SVC()
        svm_model.fit(training_input, training_output)
        prediction_svm = svm_model.predict(test_input)
        self.svmm = metrics.accuracy_score(prediction_svm, test_output)

    def k_nearest_neighbors(self, training_input, training_output, test_input, test_output):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(training_input, training_output)
        prediction_knn = knn_model.predict(test_input)
        self.knn = metrics.accuracy_score(prediction_knn, test_output)

    def decision_tree_classifier(self, training_input, training_output, test_input, test_output):
        dtc_model = DecisionTreeClassifier()
        dtc_model.fit(training_input, training_output)
        prediction_dtc = dtc_model.predict(test_input)
        self.dtc = metrics.accuracy_score(prediction_dtc, test_output)

    def gaussian_nb(self, training_input, training_output, test_input, test_output):
        gnb_model = GaussianNB()
        gnb_model.fit(training_input, training_output)
        prediction_gnb = gnb_model.predict(test_input)
        self.gnb = metrics.accuracy_score(prediction_gnb, test_output)

    def logistic_regression(self, training_input, training_output, test_input, test_output):
        lr_model = LogisticRegression()
        lr_model.fit(training_input, training_output)
        prediction_lr = lr_model.predict(test_input)
        self.lr = metrics.accuracy_score(prediction_lr, test_output)
        lr_cr = classification_report(prediction_lr, test_output)
        # print(f'\tClassification report: {lr_cr}%')

    def k_means(self, training_input, training_output, test_input, test_output):
        km_model = KMeans(n_clusters=2)
        km_model.fit(training_input, training_output)
        prediction_km = km_model.predict(test_input)
        self.km = metrics.accuracy_score(prediction_km, test_output)

    def results(self):
        model_results = pd.DataFrame({
            'Model': ['Random Forest Classifier', 'Support-vector machine', 'K Nearest Neighbors',
                      'Decision Tree Classifier', 'GaussianNB', 'Logistic Regression', 'KMeans'],
            'Rezultat': [round(self.rfc, 2), round(self.svmm, 2), round(self.knn, 2), round(self.dtc, 2),
                         round(self.gnb, 2), round(self.lr, 2), round(self.km, 2)]})
        model_results.sort_values(by='Rezultat', ascending=False)
        return model_results
