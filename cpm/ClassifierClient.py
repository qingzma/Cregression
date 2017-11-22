from __future__ import print_function,division
import ClipperClient as cc
import Data as dt
import  numpy as np
from sklearn import svm
from sklearn.svm import SVC
from collections import Counter
from datetime import datetime

#bool_time_term_in_classifier = False
num_model_in_classifier = 10
app_names_for_classifier = None
b_default_prediction_influence = True

def get_predictions_to_build_classifier(training_data_classifier):
    answers = []
    #print(training_data_classifier.__len__)
    for i in range(len(cc.app_names_deployed)):
        app_name = cc.app_names_deployed[i]
        answer_i = cc.get_predictions(app_name, training_data_classifier)
        answers.append(answer_i)

    return answers
'''
def enable_time_term_in_classifier(b =True):
    global bool_time_term_in_classifier
    bool_time_term_in_classifier = b
'''

def init_classifier_training_values(predictions, model_selection_index=None, factor=1):
    global app_names_for_classifier
    if model_selection_index != None:
        predictions = [ predictions[i] for i in model_selection_index]
        app_names_for_classifier = [cc.app_names_deployed[i] for i in model_selection_index]
    else:
        app_names_for_classifier = cc.app_names_deployed

    #-----------------

    #------------------
    dimension = len(predictions)
    global num_model_in_classifier
    num_model_in_classifier = dimension
    rankings = []
    minimum_errors = []
    num_predictions = len(predictions[0].predictions)

    for i in range(num_predictions):
        values =[]
        for j in range(dimension):
            element_of_values = factor*abs(predictions[j].predictions[i]-predictions[j].labels[i]) + \
                                (1-factor)*predictions[j].latency
            proportion_of_default_prediction = predictions[j].num_defaults/(predictions[j].num_defaults+ \
                                                                            predictions[j].num_success)
            if b_default_prediction_influence:
                element_of_values = element_of_values * (1+proportion_of_default_prediction)
            values.append(element_of_values)
            #print(values)

        rankings.append(values.index(min(values)))
        minimum_errors.append(min(values))


    model_counts = []
    for i in range(dimension):
        model_counts.append(rankings.count(i))
    model_counts_str = np.array_str(np.asarray(model_counts))

    print("Queries are classified into %d categories:  " % (dimension))
    print("Counts are: %s." % (model_counts_str))

    return rankings, minimum_errors

def build_classifier(training_data_classifier, y_classifier, C=10):
    start = datetime.now()
    distribution = Counter(y_classifier)

    if len(distribution.keys()) == 1:
        class classifier1:
            def predict(self, x):
                return [y_classifier[0]]
        classifier = classifier1()
        print("Warning: Only one best model is found! New query will only go to this prediction model!")
        print("To use more models, please change the facotr of time term to be greater than 0.")
    else:

        classifier = svm.LinearSVC(C=C)
        classifier.fit(training_data_classifier.features, y_classifier)

    end = datetime.now()
    print("Total time spent: %.4f s." % (end - start).total_seconds())
    return classifier

def build_classifier_rbf(training_data_classifier, y_classifier, C=10):
    start = datetime.now()
    distribution = Counter(y_classifier)

    if len(distribution.keys()) == 1:
        class classifier1:
            def predict(self, x):
                return [y_classifier[0]]
        classifier = classifier1()
        print("Warning: Only one best model is found! New query will only go to this prediction model!")
        print("To use more models, please change the facotr of time term to be greater than 0.")
    else:
        classifier = SVC(C=C,kernel='rbf')
        classifier.fit(training_data_classifier.features, y_classifier)

    end = datetime.now()
    print("Total time spent: %.4f s." % (end - start).total_seconds())

    return classifier


def get_cluster_points(model_number,y_classifier,training_points_features):
    results = []
    for i,element in enumerate(y_classifier):
        if element == model_number:
            results.append(np.asarray(training_points_features[i]))
    return np.asarray(results)


if __name__ == "__main__":
    resultss = [0, 1, 2, 6, 4]
    index = [0, 1, 3]
    init_classifier_training_values(resultss,index)
