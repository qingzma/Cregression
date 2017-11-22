#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import Data as dt

import random
import socket
import time
import  numpy as np
import os
import sys
import requests
import json

import findspark
findspark.init()
import pyspark

import subprocess32 as subprocess
import pprint

#from pyspark.mllib.classification import LogisticRegressionWithSGD
#from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LinearRegressionWithSGD
#from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
#from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession

from datetime import datetime
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

from clipper_admin import Clipper
import ClassifierClient




headers = {'Content-type': 'application/json'}
preds = "-1.0"
container_number = 2
input_type = "floats"

app_name0 = "sklearn_linear_regression"
model_name0 = "sklearn_linear_model"

app_name1 = "sklearn_poly_regression"
model_name1 = "sklearn_poly_model"

app_name2 = "sklearn_knn_regression"
model_name2 = "sklearn_knn_model"

app_name3 = "sklearn_svr_rbf_regression"
model_name3 = "sklearn_svr_rbf_model"

app_name4 = "mllib_regression"
model_name4 = "mllib_lrm_SGD_model"

app_name5 = "sklearn_gaussian_process_regression"
model_name5 = "sklearn_gaussian_process_model"

app_name6 = "sklearn_ensemble_adaboost_regression"
model_name6 = "sklearn_ensemble_adaboost_model"

app_name7 = "sklearn_ensemble_gradient_tree_boosting_regression"
model_name7 = "sklearn_ensemble_gradient_tree_boosting_model"

app_name8 = "sklearn_decision_tree_regression"
model_name8 = "sklearn_decision_tree_model"

#app_names = [app_name0, app_name1, app_name2, app_name3 ]   #,app_name4, app_name5]
app_names_deployed = [app_name0, app_name1, app_name2, app_name3,app_name8]


#-------------------------------------------------------------------------------------------
class BenchmarkException(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)

#-------------------------------------------------------------------------------------------
# range of ports where available ports can be found
PORT_RANGE = [34256, 40000]

def find_unbound_port():
    """
    Returns an unbound port number on 127.0.0.1.
    """
    while True:
        port = random.randint(*PORT_RANGE)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except socket.error:
            print("randomly generated port %d is bound. Trying again." % port)


def init_clipper():
    clipper = Clipper("localhost", redis_port=find_unbound_port())
    clipper.stop_all()
    clipper.start()
    time.sleep(1)
    return clipper

def predict(spark, model, xs):
    return [str(model.predict(x)) for x in xs]

def _test_deployed_model(app, version,training_data):
    #time.sleep(25)
    #num_preds = 1
    #num_defaults = 0
    #for i in range(num_preds):
    response = requests.post(
        "http://localhost:1337/%s/predict" % app,
        headers=headers,
        data=json.dumps({
            'input': list(get_test_point(training_data))
        }))
    result = response.json()
    print(result)

def get_test_point(training_data):
    #print(list(training_data_model.features[1]))
    #print(training_data)
    #print(training_data.features)
    #print(training_data.features[1])
    #global sklearn_poly_model
    #sklearn_poly_model = train_sklearn_poly_regression(training_data)
    #print(sklearn_poly_predict_fn(training_data.features[1]))
    #print(list(training_data[1].features))
    return  training_data.features[1]   #[13.0,1073.0, 0.663]

def get_predictions(app, xs):
    print("Start querying to %s." % (app))
    answer = dt.PredictionSummary()
    # xs = [[1.0 2.0],[2.0 3.0]]
    num_defaults = 0
    num_success = 0
    results = []
    # print(len(xs))
    start = datetime.now()
    for element in xs.features:
        #print(element)
        response = requests.post(
            "http://localhost:1337/%s/predict" % app,
            headers=headers,
            data=json.dumps({
                'input': list(element)
            }))
        result = response.json()

        results.append(result["output"])
        answer.modelID.append(0)

        if response.status_code == requests.codes.ok and result["default"] == False:
            num_success += 1
            answer.status.append(1)
            # results.append(result["output"])
        elif response.status_code != requests.codes.ok:
            print(result)
            #       raise BenchmarkException(response.text)
            num_defaults += 1
            answer.status.append(0)
            # results.append(result["output"])
        else:
            num_defaults += 1
            answer.status.append(0)
            # results.append(result["output"])
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0 / len(xs)
    throughput = 1000 / latency
    print("Finish %d queries, average latency is %f ms, throughput is %f /s. " % (len(xs), latency, throughput))
    if num_defaults > 0:
        print("Warning: %d of %d quries returns the default value -1." % (num_defaults,len(xs)))

    print("Total time spent: %.2f s." % (end - start).total_seconds())

    print()

    answer.predictions = results
    answer.latency = latency
    answer.throughput = throughput
    answer.labels = xs.labels
    answer.num_defaults = num_defaults
    answer.num_success = num_success
    answer.model_name = app



    return answer

def get_predictions_from_models_for_testing(training_data):
    answers = []
    #print(training_data_classifier.__len__)
    for i in range(len(ClassifierClient.app_names_for_classifier)):
        app_name = ClassifierClient.app_names_for_classifier[i]
        answer_i = get_predictions(app_name, training_data)
        answers.append(answer_i)

    return answers


def get_classified_predictions(classifier, xs):
    print("Start querying to Classified Query Client.")
    answer = dt.PredictionSummary()
    # xs = [[1.0 2.0],[2.0 3.0]]
    num_defaults = 0
    num_success = 0
    results = []
    time_classifier = []
    # print(len(xs))
    start = datetime.now()
    for element in xs.features:
        #print(element)

        start_i = datetime.now()
        model_number = classifier.predict(element.reshape(1, -1))
        end_i = datetime.now()
        time_classifier.append((end_i - start_i).total_seconds() * 1000.0)
        answer.modelID.append(model_number[0])

        response = requests.post(
            "http://localhost:1337/%s/predict" % ClassifierClient.app_names_for_classifier[model_number[0]],
            headers=headers,
            data=json.dumps({
                'input': list(element)
            }))
        result = response.json()

        results.append(result["output"])


        if response.status_code == requests.codes.ok and result["default"] == False:
            num_success += 1
            answer.status.append(1)
            # results.append(result["output"])
        elif response.status_code != requests.codes.ok:
            print(result)
            #       raise BenchmarkException(response.text)
            num_defaults += 1
            answer.status.append(0)
            # results.append(result["output"])
        else:
            num_defaults += 1
            answer.status.append(0)
            # results.append(result["output"])
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0 / len(xs)
    throughput = 1000 / latency

    print("Finish %d queries, average latency is %f ms, throughput is %f /s. " % (len(xs), latency, throughput))
    print("Average time spent on the classifier is %f ms." % (sum(time_classifier) / float(len(time_classifier))))

    if num_defaults > 0:
        print("Warning: %d of %d quries returns the default value -1." % (num_defaults,len(xs)))

    print("Total time spent: %.2f s." % (end - start).total_seconds())
    print()

    answer.predictions = results
    answer.latency = latency
    answer.throughput = throughput
    answer.labels = xs.labels
    answer.num_success = num_success
    answer.num_defaults = num_defaults
    answer.model_name = "classified model"

    #print(answer.modelID)


    # print statistics for the queries
    model_counts = []
    for i in range(ClassifierClient.num_model_in_classifier):
        model_counts.append(answer.modelID.count(i))
    model_counts_str = np.array_str(np.asarray(model_counts))

    print("Queries are classified into %d categories:  " % (ClassifierClient.num_model_in_classifier))
    print("Counts are: %s." % (model_counts_str))


    return answer



# parse the data in a line
def parsePoint(line):
    values = [float(x) for x in line]
    return LabeledPoint(values[3],values[0:3])

#return the traing data and tesing data, RDD values
def load_data(sc):
    data = sc.textFile("OnlineNewsPopularity.csv")
    filteredData=data.map(lambda x: x.replace(',', ' ')).map(lambda x: x.split()).map(lambda x:(x[2],x[3],x[4],x[6]))
    parsedData =  filteredData.map(parsePoint)
    query_training_data,trainingData, testingData = parsedData.randomSplit([0.3, 0.3,0.4])
    return query_training_data, trainingData, testingData

#-------------------------------------------------------------------------------------------------
def train_mllib_linear_regression_withSGD(trainingDataRDD):
    return LinearRegressionWithSGD.train(trainingDataRDD,iterations=500, step=0.0000000000000001,convergenceTol=0.0001,intercept=True) #,initialWeights=np.array[1.0])





#--------------------------------------------------------------------------------------------
""" Deploy and link function"""
def deploy_and_link_python_model(app_name,
                                 model_name,
                                 clipper,
                                 version,
                                 input_type,
                                 predict_fn):
    ## predict_fn is based on the model defined before this function is called.
    ## Thus, it is necessary to define the model first, then create the prediction function.
    clipper.register_application(app_name, input_type, "-1.0", 10000000)
    # sklearn_linear_model = train_sklearn_linear_regression(training_data_RDD)

    #time.sleep(5)

    clipper.deploy_predict_function(model_name, version, predict_fn, input_type, num_containers=container_number)

    bool_deployed = clipper.link_model_to_app(app_name, model_name)

    return bool_deployed

def deploy_model_spark(app_name,
                          model_name,
                          sc,
                          clipper,
                          model,
                          version,
                          input_type,
                          link_model=True,
                          predict_fn=predict):
    bool_deployed = clipper.deploy_pyspark_model(model_name, version, predict_fn, model, sc,
                                 input_type, num_containers= container_number)
    #time.sleep(5)

    if link_model:
        clipper.link_model_to_app(app_name, model_name)
    #    time.sleep(5)

    #_test_deployed_model(app_name, version)
    return bool_deployed



def deploy_model_sklearn_linear_regression(training_data,clipper,version=1):
    def train_sklearn_linear_regression(trainingData):

        X = trainingData.features
        y = trainingData.labels
        # print(X)
        # print(y)
        reg = linear_model.LinearRegression()
        reg.fit(X, y)
        return reg

    def sklearn_lr_predict_fn(inputs):
        return sklearn_linear_model.predict(inputs)
    #global sklearn_linear_model
    sklearn_linear_model = train_sklearn_linear_regression(training_data)

    bool_deployed = deploy_and_link_python_model(app_name0, model_name0, clipper, version, input_type, sklearn_lr_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed "+app_name0)
    else:
        print("Fail to deploy "+app_name0)

def deploy_model_sklearn_poly_regression(training_data,clipper,version=1):
    def train_sklearn_poly_regression(trainingData):
        X = trainingData.features
        y = trainingData.labels
        model = make_pipeline(PolynomialFeatures(2), Ridge())
        return model.fit(X, y)

    def sklearn_poly_predict_fn(inputs):
        return sklearn_poly_model.predict(inputs)
    #global sklearn_poly_model
    sklearn_poly_model = train_sklearn_poly_regression(training_data)
    bool_deployed = deploy_and_link_python_model(app_name1, model_name1, clipper, version, input_type, sklearn_poly_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed "+app_name1)
    else:
        print("Fail to deploy "+app_name1)

def deploy_model_sklearn_knn_regression(training_data,clipper,version=1):
    def train_sklearn_knn_regression(trainingData):
        n_neighbors = 5
        weights = 'distance'  # or 'uniform'
        X = trainingData.features
        y = trainingData.labels
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        knn.fit(X, y)
        return knn

    def sklearn_knn_predict_fn(inputs):
        return sklearn_knn_model.predict(inputs)
    #global sklearn_knn_model
    sklearn_knn_model = train_sklearn_knn_regression(training_data)
    bool_deployed = deploy_and_link_python_model(app_name2, model_name2, clipper, version, input_type, sklearn_knn_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed "+app_name2)
    else:
        print("Fail to deploy "+app_name2)

def deploy_model_sklearn_svr_rbf_regression(training_data,clipper,version=1):
    def train_sklearn_rbf_regression(trainingData):
        X = trainingData.features
        y = trainingData.labels
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1,cache_size=10000)
        return svr_rbf.fit(X, y)

    def sklearn_rbf_predict_fn(inputs):
        return sklearn_rbf_model.predict(inputs)
    #global sklearn_rbf_model
    sklearn_rbf_model = train_sklearn_rbf_regression(training_data)
    bool_deployed = deploy_and_link_python_model(app_name3, model_name3, clipper, version, input_type, sklearn_rbf_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed "+app_name3)
    else:
        print("Fail to deploy "+app_name3)

def deploy_model_mllib_regression(training_data_model, clipper,version=1):
    try:
        spark = SparkSession \
            .builder \
            .appName(app_name1) \
            .getOrCreate()
        sc = spark.sparkContext
        clipper.register_application(app_name4, input_type, "-1.0",10000000)

        training_data_RDD = training_data_model.toRDD(spark)
        mllib_model = train_mllib_linear_regression_withSGD(training_data_RDD)
        bool_deployed = deploy_model_spark(app_name4, model_name4, sc, clipper, mllib_model, version, input_type, True)
    except BenchmarkException as e:
        print(e)
        clipper.stop_all()
        spark.stop()
        sys.exit(1)
    else:
        #            spark.stop()
        #            clipper.stop_all()
        print("Successfully deployed mllib linear regression model! ")
    if bool_deployed:
        print("Sucessfully deployed "+app_name4)
    else:
        print("Fail to deploy "+app_name4)
    return spark

def deploy_model_sklearn_gaussion_process_regression(training_data,clipper,version=1):
    def train_sklearn_gp_regression(trainingData):
        X = trainingData.features
        y = trainingData.labels
        # Instanciate a Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        return gp.fit(X, y)

    def sklearn_gp_predict_fn(inputs):
        return sklearn_gp_model.predict(inputs)
    #global sklearn_gp_models
    sklearn_gp_model = train_sklearn_gp_regression(training_data)
    bool_deployed = deploy_and_link_python_model(app_name5, model_name5, clipper, version, input_type, sklearn_gp_predict_fn)

    if bool_deployed:
        print("Sucessfully deployed "+app_name5)
    else:
        print("Fail to deploy "+app_name5)

def deploy_model_sklearn_ensemble_adaboost(training_data, clipper, version=1):
    def train_sklearn_ensemble_adaboost(trainingData):

        X = trainingData.features
        y = trainingData.labels
        # print(X)
        # print(y)
        #reg = linear_model.LinearRegression()
        start = datetime.now()

        reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                   n_estimators=300 ) ##, random_state=rng)
        reg.fit(X, y)
        end = datetime.now()
        print("Total time spent: %.2f s." % (end - start).total_seconds())

        return reg

    def sklearn_ensemble_adaboost_predict_fn(inputs):
        return sklearn_adaboost_model.predict(inputs)

    # global sklearn_linear_model
    sklearn_adaboost_model = train_sklearn_ensemble_adaboost(training_data)

    bool_deployed = deploy_and_link_python_model(app_name6, model_name6, clipper, version, input_type,
                                                 sklearn_ensemble_adaboost_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed " + app_name6)
    else:
        print("Fail to deploy " + app_name6)

def deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data, clipper, version=1):
    def train_sklearn_ensemble_gradient_tree_boosting(trainingData):

        X = trainingData.features
        y = trainingData.labels
        # print(X)
        # print(y)
        # reg = linear_model.LinearRegression()
        start = datetime.now()

        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
        reg.fit(X, y)

        end = datetime.now()
        print("Total time spent: %.2f s." % (end - start).total_seconds())

        return reg

    def sklearn_ensemble_gradient_tree_boosting_predict_fn(inputs):
        return sklearn_gradient_tree_boosting_model.predict(inputs)

    # global sklearn_linear_model
    sklearn_gradient_tree_boosting_model = train_sklearn_ensemble_gradient_tree_boosting(training_data)

    bool_deployed = deploy_and_link_python_model(app_name7, model_name7, clipper, version, input_type,
                                                 sklearn_ensemble_gradient_tree_boosting_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed " + app_name7)
    else:
        print("Fail to deploy " + app_name7)

def deploy_model_sklearn_decision_tree_regression(training_data, clipper, version=1):
    def train_sklearn_decision_tree_regression(trainingData):

        X = trainingData.features
        y = trainingData.labels
        # print(X)
        # print(y)
        # reg = linear_model.LinearRegression()
        reg = DecisionTreeRegressor(max_depth=4)
        reg.fit(X, y)
        return reg

    def sklearn_ensemble_decision_tree_predict_fn(inputs):
        return sklearn_decision_tree_model.predict(inputs)

    # global sklearn_linear_model
    sklearn_decision_tree_model = train_sklearn_decision_tree_regression(training_data)

    bool_deployed = deploy_and_link_python_model(app_name8, model_name8, clipper, version, input_type,
                                                 sklearn_ensemble_decision_tree_predict_fn)
    if bool_deployed:
        print("Sucessfully deployed " + app_name8)
    else:
        print("Fail to deploy " + app_name8)
#-------------------------------------------------------------------------------------------------

def deploy_all_models(training_data, apps_to_deploy=None ):
    # start clipper
    version = 1
    clipper = init_clipper()

    global app_names_deployed
    if apps_to_deploy == None:
        app_names_deployed = [app_name0, app_name1, app_name2, app_name3,app_name8]
    else:
        app_names_deployed = apps_to_deploy


    deploy_model_sklearn_linear_regression(training_data,clipper,version)
    deploy_model_sklearn_poly_regression(training_data, clipper, version)
    deploy_model_sklearn_knn_regression(training_data, clipper, version)
    deploy_model_sklearn_svr_rbf_regression(training_data, clipper, version)
    deploy_model_sklearn_decision_tree_regression(training_data, clipper, version)


    #####deploy_model_mllib_regression(training_data, clipper, version)
    #####deploy_model_sklearn_gaussion_process_regression(training_data, clipper, version)

    deploy_model_sklearn_ensemble_adaboost(training_data, clipper, version)
    deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data,clipper,version)





    return clipper



def stop_clipper(clipper,spark=None):
    clipper.stop_all()
    if spark != None:
        spark.stop()

def __get_app_names__():
    app_names =[app_name0, app_name1,app_name2, app_name3]
    return app_names

def set_app_names_deployed(names):
    global app_names_deployed
    app_names_deployed = names
    return True

def get_app_names_deployed():
    global app_names_deployed
    return app_names_deployed
# -----------------------------------------------------------------------------------------------
'''
def store_training_data(training_data_model1, training_data_classifier1, testing_data1):
    training_data_model = training_data_model1
    training_data_classifier = training_data_classifier1
    testing_data = testing_data1

def get_training_data():
    return training_data_model, training_data_classifier, testing_data
'''
#-------------------------------------------------------------------------------------------------
if __name__ =="__main__":
    version = 1

    # load the data
    fields = ['n_tokens_title','n_tokens_content','n_non_stop_unique_tokens', 'n_unique_tokens']
    y_column = 2    # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("OnlineNewsPopularity1.csv",fields,y_column)
    #global training_data_model
    training_data_model, training_data_classifier, testing_data = dt.split_data(data)




    # start clipper
    #clipper = init_clipper()

    # deploy mllib model
    #spark = deploy_model_mllib_regression(training_data_model,clipper,version)

'''
    deploy_model_sklearn_linear_regression(training_data_model,clipper,version)

    print("sleep for 5 seconds.")
    time.sleep(5)
    _test_deployed_model(app_name0, version,training_data)
'''


    # Stop clipper and spark
    #stop_clipper(clipper,spark)




