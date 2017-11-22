# #!/usr/bin/env python
# # coding=utf-8
# from __future__ import print_function, division
# import Data as dt
#
# #import random
# #import socket
# import time
# import  numpy as np
# import os
# import sys
# #import requests
# #import json
# from collections import Counter
# from sklearn import svm
# from sklearn.svm import SVC
#
# # Path for spark source folder
# os.environ['SPARK_HOME']="/home/u1796377/Program/spark-2.1.0-bin-hadoop2.7"
#
# # Append pyspark  to Python Path
# sys.path.append("/home/u1796377/Program/spark-2.1.0-bin-hadoop2.7")
#
# import findspark
# findspark.init()
# import pyspark
#
# import subprocess32 as subprocess
# import pprint
#
# #from pyspark.mllib.classification import LogisticRegressionWithSGD
# #from pyspark.mllib.classification import SVMWithSGD
# from pyspark.mllib.regression import LinearRegressionWithSGD
# #from pyspark.mllib.tree import RandomForest
# from pyspark.mllib.regression import LabeledPoint
# #from pyspark.ml.regression import GeneralizedLinearRegression
# from pyspark.sql import SparkSession
#
# from datetime import datetime
# from sklearn import linear_model
# from sklearn import neighbors
# from sklearn import svm
# from sklearn.svm import SVR
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.tree import DecisionTreeRegressor
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# #%matplotlib inline
#
# #from clipper_admin import Clipper
# #import ClassifierClient
#
#
# # package for different classifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#
#
#
# headers = {'Content-type': 'application/json'}
# preds = "-1.0"
# container_number = 2
# input_type = "floats"
#
# app_name0 = "sklearn_linear_regression"
# model_name0 = "sklearn_linear_model"
#
# app_name1 = "sklearn_poly_regression"
# model_name1 = "sklearn_poly_model"
#
# app_name2 = "sklearn_knn_regression"
# model_name2 = "sklearn_knn_model"
#
# app_name3 = "sklearn_svr_rbf_regression"
# model_name3 = "sklearn_svr_rbf_model"
#
# app_name4 = "mllib_regression"
# model_name4 = "mllib_lrm_SGD_model"
#
# app_name5 = "sklearn_gaussian_process_regression"
# model_name5 = "sklearn_gaussian_process_model"
#
# app_name6 = "sklearn_ensemble_adaboost_regression"
# model_name6 = "sklearn_ensemble_adaboost_model"
#
# app_name7 = "sklearn_ensemble_gradient_tree_boosting_regression"
# model_name7 = "sklearn_ensemble_gradient_tree_boosting_model"
#
# app_name8 = "sklearn_decision_tree_regression"
# model_name8 = "sklearn_decision_tree_model"
#
# #app_names = [app_name0, app_name1, app_name2, app_name3 ]   #,app_name4, app_name5]
# app_names_deployed = []
#
# apps_deployed = []
# index_of_models_in_classifier = []
#
# #bool_time_term_in_classifier = False
# num_model_in_classifier = 10
# app_names_for_classifier = None
# apps_for_classifier = None
# ensemble_method_names=[app_name6,app_name7]
#
#
#
#
#
# #-------------------------------------------------------------------------------------------
# class BenchmarkException(Exception):
#     def __init__(self, value):
#         self.parameter = value
#
#     def __str__(self):
#         return repr(self.parameter)
#
# #-------------------------------------------------------------------------------------------
#
#
# def predict(spark, model, xs):
#     return [str(model.predict(x)) for x in xs]
#
# def _test_deployed_model(model,training_data):
#     '''
#     response = requests.post(
#         "http://localhost:1337/%s/predict" % app,
#         headers=headers,
#         data=json.dumps({
#             'input': list(get_test_point(training_data))
#         }))
#     result = response.json()
#     '''
#     result = model.predict(training_data)
#     print(result)
#
# def get_test_point(training_data):
#     #print(list(training_data_model.features[1]))
#     #print(training_data)
#     #print(training_data.features)
#     #print(training_data.features[1])
#     #global sklearn_poly_model
#     #sklearn_poly_model = train_sklearn_poly_regression(training_data)
#     #print(sklearn_poly_predict_fn(training_data.features[1]))
#     #print(list(training_data[1].features))
#     return  training_data.features[1]   #[13.0,1073.0, 0.663]
#
# def get_predictions(app, xs):
#     global apps_deployed
#     global app_names_deployed
#     print("Start querying to %s." % (app_names_deployed[apps_deployed.index(app)]))
#     answer = dt.PredictionSummary()
#     # xs = [[1.0 2.0],[2.0 3.0]]
#     num_defaults = 0
#     num_success = len(xs.features)
#     results = []
#     # print(len(xs))
#     start = datetime.now()
#     for element in xs.features:
#         #print(element)
#
#         #print(element)
#         results.append(app.predict([element])[0])
#         answer.status.append(1)
#
#     end = datetime.now()
#     latency = (end - start).total_seconds() * 1000.0 / len(xs)
#     #throughput = 1000 / latency
#     print("Finish %d queries, average latency is %f ms. " % (len(xs), latency))
#     if num_defaults > 0:
#         print("Warning: %d of %d quries returns the default value -1." % (num_defaults,len(xs)))
#
#     print("Total time spent: %.2f s." % (end - start).total_seconds())
#
#     print()
#
#     answer.predictions = results
#     answer.latency = latency
#     #answer.throughput = throughput
#     answer.labels = xs.labels
#     answer.num_defaults = num_defaults
#     answer.num_success = num_success
#     answer.model_name = app_names_deployed[apps_deployed.index(app)]
#     answer.features = xs.features
#     answer.headers =xs.headers
#     answer.time_total = (end - start).total_seconds()
#     answer.num_of_instances = len(xs)
#
#
#
#     return answer
#
# def get_predictions_from_models_for_testing(training_data):
#     answers = []
#     global app_names_for_classifier
#     global apps_for_classifier
#     #print(training_data_classifier.__len__)
#     for i in range(len(apps_for_classifier)):
#         app_i = apps_for_classifier[i]
#         answer_i = get_predictions(app_i, training_data)
#         answers.append(answer_i)
#
#     return answers
#
#
# def get_classified_predictions(classifier, xs):
#     global apps_deployed
#     print("Start querying to Classified Prediction System.")
#     answer = dt.PredictionSummary()
#     # xs = [[1.0 2.0],[2.0 3.0]]
#     num_defaults = 0
#     num_success = len(xs)
#     results = []
#     time_classifier = []
#     # print(len(xs))
#     start = datetime.now()
#     for element in xs.features:
#         #print(element)
#
#         start_i = datetime.now()
#         model_number = classifier.predict(element.reshape(1, -1))
#         end_i = datetime.now()
#         time_classifier.append((end_i - start_i).total_seconds() * 1000.0)
#         answer.modelID.append(model_number[0])
#
#         #response = requests.post(
#         #    "http://localhost:1337/%s/predict" % ClassifierClient.app_names_for_classifier[model_number[0]],
#         #    headers=headers,
#         #    data=json.dumps({
#         #        'input': list(element)
#         #    }))
#         #result = response.json()
#
#         #results.append(result["output"])
#         answer.status.append(1)
#         #print(model_number[0])
#         #print(len(apps_deployed))
#         #print(element)
#         #print(list(element))
#         #print(np.array(list(element)).reshape(1,-1))
#
#         #print()
#         value_tmp=apps_deployed[model_number[0]].predict(np.array(list(element)).reshape(1,-1))
#         value=value_tmp[0]
#         results.append(value)
#
#
#
#     end = datetime.now()
#     latency = (end - start).total_seconds() * 1000.0 / len(xs)
#     #throughput = 1000 / latency
#
#     print("Finish %d queries, average latency is %f ms. " % (len(xs), latency))
#     print("Average time spent on the classifier is %f ms." % (sum(time_classifier) / float(len(time_classifier))))
#
#     if num_defaults > 0:
#         print("Warning: %d of %d quries returns the default value -1." % (num_defaults,len(xs)))
#
#     print("Total time spent: %.2f s." % (end - start).total_seconds())
#     print()
#
#     answer.predictions = results
#     answer.latency = latency
#     #answer.throughput = throughput
#     answer.labels = xs.labels
#     answer.num_success = num_success
#     answer.num_defaults = num_defaults
#     answer.model_name = "classified model"
#     answer.features = xs.features
#     answer.headers =xs.headers
#     answer.time_total= (end - start).total_seconds()
#     answer.time_query_execution_on_classifier=(sum(time_classifier) / float(len(time_classifier)))
#     answer.num_of_instances=len(time_classifier)
#     #print(answer.modelID)
#
#
#     # print statistics for the queries
#     model_counts = []
#     for i in range(num_model_in_classifier):
#         model_counts.append(answer.modelID.count(i))
#     model_counts_str = np.array_str(np.asarray(model_counts))
#
#     print("Queries are classified into %d categories:  " % (num_model_in_classifier))
#     print("Counts are: %s." % (model_counts_str))
#
#
#     return answer
#
#
# def get_classified_prediction(classifier, x):
#     global apps_deployed
#     X=[x]
#     model_number = classifier.predict(X)
#     return apps_deployed[model_number[0]].predict(np.array(x).reshape(1, -1))[0]
#
# # parse the data in a line
# def parsePoint(line):
#     values = [float(x) for x in line]
#     return LabeledPoint(values[3],values[0:3])
#
# #return the traing data and tesing data, RDD values
# def load_data(sc):
#     data = sc.textFile("OnlineNewsPopularity.csv")
#     filteredData=data.map(lambda x: x.replace(',', ' ')).map(lambda x: x.split()).map(lambda x:(x[2],x[3],x[4],x[6]))
#     parsedData =  filteredData.map(parsePoint)
#     query_training_data,trainingData, testingData = parsedData.randomSplit([0.3, 0.3,0.4])
#     return query_training_data, trainingData, testingData
#
# #-------------------------------------------------------------------------------------------------
# def train_mllib_linear_regression_withSGD(trainingDataRDD):
#     return LinearRegressionWithSGD.train(trainingDataRDD,iterations=500, step=0.0000000000000001,convergenceTol=0.0001,intercept=True) #,initialWeights=np.array[1.0])
#
#
#
#
#
# #--------------------------------------------------------------------------------------------
# """ Deploy and link function"""
# def deploy_and_link_python_model(app_name,
#                                  model_name,
#                                  clipper,
#                                  version,
#                                  input_type,
#                                  predict_fn):
#     ## predict_fn is based on the model defined before this function is called.
#     ## Thus, it is necessary to define the model first, then create the prediction function.
#     clipper.register_application(app_name, input_type, "-1.0", 10000000)
#     # sklearn_linear_model = train_sklearn_linear_regression(training_data_RDD)
#
#     #time.sleep(5)
#
#     clipper.deploy_predict_function(model_name, version, predict_fn, input_type, num_containers=container_number)
#
#     bool_deployed = clipper.link_model_to_app(app_name, model_name)
#
#     return bool_deployed
#
# def deploy_model_spark(app_name,
#                           model_name,
#                           sc,
#                           clipper,
#                           model,
#                           version,
#                           input_type,
#                           link_model=True,
#                           predict_fn=predict):
#     bool_deployed = clipper.deploy_pyspark_model(model_name, version, predict_fn, model, sc,
#                                  input_type, num_containers= container_number)
#     #time.sleep(5)
#
#     if link_model:
#         clipper.link_model_to_app(app_name, model_name)
#     #    time.sleep(5)
#
#     #_test_deployed_model(app_name, version)
#     return bool_deployed
#
#
#
# def deploy_model_sklearn_linear_regression(training_data):
#     def train_sklearn_linear_regression(trainingData):
#         start = datetime.now()
#         X = trainingData.features
#         y = trainingData.labels
#         # print(X)
#         # print(y)
#         reg = linear_model.LinearRegression()
#         reg.fit(X, y)
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#         return reg
#
#     def sklearn_lr_predict_fn(inputs):
#         return sklearn_linear_model.predict(inputs)
#     #global sklearn_linear_model
#     sklearn_linear_model = train_sklearn_linear_regression(training_data)
#     print("Sucessfully deployed "+app_name0)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_linear_model)
#     global  app_names_deployed
#     app_names_deployed.append(app_name0)
#
#     return sklearn_linear_model
#
#
# def deploy_model_sklearn_poly_regression(training_data):
#     def train_sklearn_poly_regression(trainingData):
#         start = datetime.now()
#         X = trainingData.features
#         y = trainingData.labels
#         model = make_pipeline(PolynomialFeatures(5), Ridge())
#
#         model.fit(X, y)
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#         return model
#
#     def sklearn_poly_predict_fn(inputs):
#         return sklearn_poly_model.predict(inputs)
#     #global sklearn_poly_model
#     sklearn_poly_model = train_sklearn_poly_regression(training_data)
#
#     print("Sucessfully deployed "+app_name1)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_poly_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name1)
#
#     return sklearn_poly_model
#
# def deploy_model_sklearn_knn_regression(training_data):
#     def train_sklearn_knn_regression(trainingData):
#         start = datetime.now()
#         n_neighbors = 5
#         weights = 'distance'  # or 'uniform'
#         X = trainingData.features
#         y = trainingData.labels
#         knn = neighbors.KNeighborsRegressor( weights=weights,n_jobs=1,n_neighbors=5)
#         knn.fit(X, y)
#
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#         return knn
#
#     def sklearn_knn_predict_fn(inputs):
#         return sklearn_knn_model.predict(inputs)
#     #global sklearn_knn_model
#     sklearn_knn_model = train_sklearn_knn_regression(training_data)
#
#     print("Sucessfully deployed "+app_name2)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_knn_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name2)
#     return sklearn_knn_model
#
# def deploy_model_sklearn_svr_rbf_regression(training_data):
#     def train_sklearn_rbf_regression(trainingData):
#         start = datetime.now()
#         X = trainingData.features
#         y = trainingData.labels
#         svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1,cache_size=10000)
#
#         svr_rbf.fit(X, y)
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#
#         return svr_rbf
#
#     def sklearn_rbf_predict_fn(inputs):
#         return sklearn_rbf_model.predict(inputs)
#     #global sklearn_rbf_model
#     sklearn_rbf_model = train_sklearn_rbf_regression(training_data)
#     print("Sucessfully deployed "+app_name3)
#     print()
#     global apps_deployed
#     apps_deployed.append(sklearn_rbf_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name3)
#     return sklearn_rbf_model
#
# def deploy_model_mllib_regression(training_data_model):
#     try:
#         spark = SparkSession \
#             .builder \
#             .appName(app_name1) \
#             .getOrCreate()
#         sc = spark.sparkContext
#         #clipper.register_application(app_name4, input_type, "-1.0",10000000)
#
#         start = datetime.now()
#         training_data_RDD = training_data_model.toRDD(spark)
#         mllib_model = train_mllib_linear_regression_withSGD(training_data_RDD)
#
#     except BenchmarkException as e:
#         print(e)
#         #clipper.stop_all()
#         spark.stop()
#         sys.exit(1)
#     else:
#         #            spark.stop()
#         #            clipper.stop_all()
#         print("Successfully deployed mllib linear regression model! ")
#     print("Sucessfully deployed "+app_name4)
#     print()
#     global apps_deployed
#     apps_deployed.append(mllib_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name4)
#
#     end = datetime.now()
#     print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#     return mllib_model
#
# def deploy_model_sklearn_gaussion_process_regression(training_data):
#     def train_sklearn_gp_regression(trainingData):
#         X = trainingData.features
#         y = trainingData.labels
#         # Instanciate a Gaussian Process model
#         kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#         gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#         # Fit to data using Maximum Likelihood Estimation of the parameters
#         start = datetime.now()
#         gp.fit(X,y)
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#         return gp
#
#     def sklearn_gp_predict_fn(inputs):
#         return sklearn_gp_model.predict(inputs)
#     #global sklearn_gp_models
#     sklearn_gp_model = train_sklearn_gp_regression(training_data)
#     print("Sucessfully deployed "+app_name5)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_gp_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name5)
#     return sklearn_gp_model
#
# def deploy_model_sklearn_ensemble_adaboost(training_data):
#     def train_sklearn_ensemble_adaboost(trainingData):
#
#         X = trainingData.features
#         y = trainingData.labels
#         # print(X)
#         # print(y)
#         #reg = linear_model.LinearRegression()
#         start = datetime.now()
#
#         reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                                    n_estimators=300 ) ##, random_state=rng)
#         reg.fit(X, y)
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#
#         return reg
#
#     def sklearn_ensemble_adaboost_predict_fn(inputs):
#         return sklearn_adaboost_model.predict(inputs)
#
#     # global sklearn_linear_model
#     sklearn_adaboost_model = train_sklearn_ensemble_adaboost(training_data)
#
#     print("Sucessfully deployed " + app_name6)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_adaboost_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name6)
#
#     return sklearn_adaboost_model
#
# def deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data):
#     def train_sklearn_ensemble_gradient_tree_boosting(trainingData):
#
#         X = trainingData.features
#         y = trainingData.labels
#         # print(X)
#         # print(y)
#         # reg = linear_model.LinearRegression()
#         start = datetime.now()
#
#         reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
#         reg.fit(X, y)
#
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#
#         return reg
#
#     def sklearn_ensemble_gradient_tree_boosting_predict_fn(inputs):
#         return sklearn_gradient_tree_boosting_model.predict(inputs)
#
#     # global sklearn_linear_model
#     sklearn_gradient_tree_boosting_model = train_sklearn_ensemble_gradient_tree_boosting(training_data)
#
#     print("Sucessfully deployed " + app_name7)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_gradient_tree_boosting_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name7)
#     return sklearn_gradient_tree_boosting_model
#
# def deploy_model_sklearn_decision_tree_regression(training_data):
#     def train_sklearn_decision_tree_regression(trainingData):
#
#         X = trainingData.features
#         y = trainingData.labels
#         # print(X)
#         # print(y)
#         # reg = linear_model.LinearRegression()
#         start = datetime.now()
#         reg = DecisionTreeRegressor(max_depth=4)
#         reg.fit(X, y)
#
#         end = datetime.now()
#         print("Time cost to train the model is : %.2f s." % (end - start).total_seconds())
#
#
#         return reg
#
#     def sklearn_decision_tree_predict_fn(inputs):
#         return sklearn_decision_tree_model.predict(inputs)
#
#     # global sklearn_linear_model
#     sklearn_decision_tree_model = train_sklearn_decision_tree_regression(training_data)
#
#     print("Sucessfully deployed " + app_name8)
#     print()
#
#     global apps_deployed
#     apps_deployed.append(sklearn_decision_tree_model)
#     global app_names_deployed
#     app_names_deployed.append(app_name8)
#     return sklearn_decision_tree_model
# #-------------------------------------------------------------------------------------------------
#
# def deploy_all_models(training_data):
#
#
#     global app_names_deployed
#     global apps_deployed
#     app_names_deployed = []
#     apps_deployed = []
#     time_cost=[]
#
#
#     time0=datetime.now()
#     deploy_model_sklearn_linear_regression(training_data)
#     time1=datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#     time0=datetime.now()
#     deploy_model_sklearn_poly_regression(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#     time0 = datetime.now()
#     deploy_model_sklearn_knn_regression(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#     #'''
#     time0 = datetime.now()
#     deploy_model_sklearn_svr_rbf_regression(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#     #'''
#
#     time0 = datetime.now()
#     deploy_model_sklearn_decision_tree_regression(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#
#     #time0 = datetime.now()
#     #deploy_model_mllib_regression(training_data)
#     #time1 = datetime.now()
#     #time_cost.append((time1 - time0).total_seconds() )
#
#     #time0 = datetime.now()
#     #deploy_model_sklearn_gaussion_process_regression(training_data)
#     #time1 = datetime.now()
#     #time_cost.append((time1 - time0).total_seconds() )
#
#
#     '''
#     time0 = datetime.now()
#     deploy_model_sklearn_ensemble_adaboost(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#     time0 = datetime.now()
#     deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#     '''
#
#
#
#     return apps_deployed,time_cost
#
# def deploy_ensemble_methods(training_data):
#     time_cost=[]
#
#     time0=datetime.now()
#     em1 = deploy_model_sklearn_ensemble_adaboost(training_data)
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#     time0 = datetime.now()
#     em2 = deploy_model_sklearn_ensemble_gradient_tree_boosting(training_data)
#     ensemble_methods_deployed = [em1,em2]
#     time1 = datetime.now()
#     time_cost.append((time1 - time0).total_seconds() )
#
#
#     #print("Deployed!")
#     return ensemble_methods_deployed,time_cost
#
#
#
# def set_app_names_deployed(names):
#     global app_names_deployed
#     app_names_deployed = names
#     return True
#
# def get_app_names_deployed():
#     global app_names_deployed
#     return app_names_deployed
# # -----------------------------------------------------------------------------------------------
#
#
# ############# the code below is a  modified version of ClassifiedClient.py, adjusted for pure python implementation.
# def get_predictions_to_build_classifier(training_data_classifier):
#     global app_names_deployed
#     global apps_deployed
#     answers = []
#     #print(training_data_classifier.__len__)
#     for i in range(len(apps_deployed)):
#         model_i = apps_deployed[i]
#         answer_i = get_predictions(model_i, training_data_classifier)
#         answers.append(answer_i)
#
#     return answers
#
#
# def init_classifier_training_values(predictions, model_selection_index=None, factor=1):
#     global app_names_for_classifier
#     global apps_for_classifier
#     global index_of_models_in_classifier
#     if model_selection_index != None:
#         predictions = [ predictions[i] for i in model_selection_index]
#         app_names_for_classifier = [app_names_deployed[i] for i in model_selection_index]
#         apps_for_classifier = [apps_deployed[i] for i in model_selection_index]
#         index_of_models_in_classifier= model_selection_index
#     else:
#         app_names_for_classifier = app_names_deployed
#         apps_for_classifier = apps_deployed
#         index_of_models_in_classifier=range(len(app_names_for_classifier))
#
#     #-----------------
#
#     #------------------
#     dimension = len(predictions)
#     global num_model_in_classifier
#     num_model_in_classifier = dimension
#     rankings = []
#     minimum_errors = []
#     num_predictions = len(predictions[0].predictions)
#
#     for i in range(num_predictions):
#         values =[]
#         for j in range(dimension):
#             element_of_values = factor*abs(predictions[j].predictions[i]-predictions[j].labels[i]) + \
#                                 (1-factor)*predictions[j].latency
#             #proportion_of_default_prediction = predictions[j].num_defaults/(predictions[j].num_defaults+ \
#             #                                                                predictions[j].num_success)
#             #if b_default_prediction_influence:
#             #    element_of_values = element_of_values * (1+proportion_of_default_prediction)
#             values.append(element_of_values)
#             #print(values)
#
#         rankings.append(values.index(min(values)))
#         minimum_errors.append(min(values))
#
#
#     model_counts = []
#     for i in range(dimension):
#         model_counts.append(rankings.count(i))
#     model_counts_str = np.array_str(np.asarray(model_counts))
#
#     print("Queries are classified into %d categories:  " % (dimension))
#     print("Counts are: %s." % (model_counts_str))
#
#     return rankings, minimum_errors
#
# def build_classifier(training_data_classifier, y_classifier, C=100):
#     start = datetime.now()
#     distribution = Counter(y_classifier)
#
#     if len(distribution.keys()) == 1:
#         class classifier1:
#             def predict(self, x):
#                 return [y_classifier[0]]
#         classifier = classifier1()
#         print("Warning: Only one best model is found! New query will only go to this prediction model!")
#         print("To use more models, please change the facotr of time term to be greater than 0.")
#     else:
#
#         classifier = svm.LinearSVC(C=C)
#         classifier.fit(training_data_classifier.features, y_classifier)
#
#     end = datetime.now()
#     print("Total time spent: %.4f s." % (end - start).total_seconds())
#     return classifier,(end - start).total_seconds()
#
# def build_classifier_rbf(training_data_classifier, y_classifier, C=1):
#     start = datetime.now()
#     distribution = Counter(y_classifier)
#
#     if len(distribution.keys()) == 1:
#         class classifier1:
#             def predict(self, x):
#                 return [y_classifier[0]]
#         classifier = classifier1()
#         print("Warning: Only one best model is found! New query will only go to this prediction model!")
#         print("To use more models, please change the facotr of time term to be greater than 0.")
#     else:
#         classifier = SVC(C=C,kernel='rbf')
#         classifier.fit(training_data_classifier.features, y_classifier)
#
#     end = datetime.now()
#     print("Total time spent: %.4f s." % (end - start).total_seconds())
#
#     return classifier,(end - start).total_seconds()
#
#
# def get_cluster_points(model_number,y_classifier,training_points_features):
#     results = []
#     for i,element in enumerate(y_classifier):
#         if element == model_number:
#             results.append(np.asarray(training_points_features[i]))
#     return np.asarray(results)
#
# '''
# if __name__ == "__main__":
#     resultss = [0, 1, 2, 6, 4]
#     index = [0, 1, 3]
#     init_classifier_training_values(resultss,index)
#
# '''
#
#
# def select_classifiers(training_data_classifier,y_classifier,testing_data):
#     global classifier_names_candidate
#     classifier_names_candidate = ["Nearest Neighbors", "Linear SVM",# "RBF SVM",
#                                   "Decision Tree", "Random Forest", "Neural Net",# "AdaBoost",
#                                   "Naive Bayes", "QDA"]
#     start = datetime.now()
#     distribution = Counter(y_classifier)
#     time_costs=[]
#
#     if len(distribution.keys()) == 1:
#         class classifier1:
#             def predict(self, x):
#                 return [y_classifier[0]]
#
#         classifier = classifier1()
#         print("Warning: Only one best model is found! New query will only go to this prediction model!")
#         print("To use more models, please change the facotr of time term to be greater than 0.")
#         time_costs.append(0.0)
#     else:
#
#
#         classifiers = [
#             KNeighborsClassifier(3),
#             svm.LinearSVC(C=100), #SVC(kernel="linear", C=0.025),
#             #SVC(gamma=2, C=1),
#             DecisionTreeClassifier(max_depth=5),
#             RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#             MLPClassifier(alpha=1),
#             #AdaBoostClassifier(),
#             GaussianNB(),
#             QuadraticDiscriminantAnalysis()]
#
#         # iterate over classifiers
#         NRMSEs = []
#         scores = []
#         for name, clf in zip(classifier_names_candidate, classifiers):
#             print("Classifier: "+ name)
#             time0=datetime.now()
#             clf.fit(training_data_classifier.features, y_classifier)
#             time1=datetime.now()
#             score = clf.score(training_data_classifier.features, y_classifier)
#             predictions_classified = get_classified_predictions(clf, testing_data)
#             NRMSEs.append(predictions_classified.NRMSE())
#             scores.append(score)
#             time_costs.append((time1-time0).seconds)
#             print("-----------------------------------------------------------")
#
#         print()
#         print()
#         print("Summary:")
#         print("NRMSEs of the classifiers:"+str(NRMSEs))
#         print("Scores of the classifiers:"+str(scores))
#
#         index = NRMSEs.index(min(NRMSEs))
#         classifier = classifiers[index]
#         print("The best classifier is: "+classifier_names_candidate[index])
#         print("The best NRMSE is: "+ str(NRMSEs[index]))
#         time_cost = time_costs[index]
#
#     return classifier, NRMSEs,time_costs,time_cost      # time cost of the best classifier
#
#
#
# def predict(x):
#     global classifier
#     return get_classified_prediction(classifier,x)
#
#
#
# def fit(training_data_model,training_data_classifier):
#     models, time_cost_to_train_base_models = deploy_all_models(training_data_model)
#
#     # get predictions to build the classifier
#     answers_for_classifier = get_predictions_to_build_classifier(training_data_classifier)
#
#     y_classifier, errors = init_classifier_training_values(answers_for_classifier,
#                                                                   # model_selection_index=index_models,
#                                                                   factor=1)
#     global classifier
#     classifier, time_cost_to_train_the_best_classifier = build_classifier(training_data_classifier, y_classifier)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #-------------------------------------------------------------------------------------------------
# if __name__ =="__main__":
#     fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
#     # fields = ['Exhaust_Vacuum','Relative_Humidity','energy_output']
#     y_column = 4  # should be the order in the input file, not in the "fields" order.
#     # data = dt.load_csv("datasets/OnlineNewsPopularity1.csv",fields,y_column)
#     data = dt.load_csv("datasets/6CCPP/Folds5x2_pp.csv", fields, y_column)
#     global training_data_model
#     training_data_model, training_data_classifier, testing_data = dt.split_data(data)
#
#     '''
#     training_data_model = training_data_model.get_before(100)
#     training_data_classifier = training_data_classifier.get_before(100)
#     testing_data = testing_data.get_before(100)
#     '''
#
#     training_data_model = training_data_model  # .get_before(500)
#     training_data_classifier = training_data_classifier  # .get_before(500)
#     testing_data = testing_data  # .get_before(500)
#
#
#     models = deploy_all_models(training_data_model)
#
#     ###app_names_deployed = [app_name0, app_name1, app_name2, app_name3, app_name8]
#
#     #answers_for_classifier = get_predictions_to_build_classifier(training_data_classifier)
#     predictions0 = get_predictions(models[0],training_data_classifier)
#     print(predictions0.predictions)
#
#
#
#
