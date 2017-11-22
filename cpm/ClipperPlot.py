#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ClassifierClient



def plot_3D(trainingDataRDD):
    X = np.array(trainingDataRDD.map(lambda p: p.features.toArray()).collect())
    y = trainingDataRDD.map(lambda p: p.label).collect()

#    %matplotlib notebook
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x0 =X[:,0]
    y0 =X[:,1]
    z0 =X[:,2]

    ax.scatter(x0, y0, z0, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # rotate the axes and update
    #ax.view_init(30, 160)
    #plt.figure(figsize=(40,30))

    plt.show()


def make_meshgrid_2D(x, y, h1=.2, h2=0.5):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    # y_min, y_max = 300, 500
    # print(x_min)
    # print(x_max)
    # print(y_min)
    # print(y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h1),
                         np.arange(y_min, y_max, h2))
    return xx, yy

def plot_contours_2D(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # print(Z)
    CS = plt.contourf(xx, yy, Z, level=[-0.5, 0.5, 1.5, 2.5, 3.5], **params)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
             for pc in CS.collections]
    #print(len(CS.collections))
    #labels = []
    #for i in range(len(CS.collections)):
    #   labels.append(ClassifierClient.app_names_for_classifier[i])
    # plt.title('Simplest default with labels')
    #plt.legend(proxy, ["range(2-3)", "range(3-4)", "range(4-6)"])
    ##plt.colorbar()





def classifier_contour_2D_1(training_data_classifier,classifier):
    XXX = np.array(training_data_classifier.features)
    X0, X1 = XXX[:, 0], XXX[:, 1]
    xx, yy = make_meshgrid_2D(X0, X1)
    plot_contours_2D(classifier, xx, yy)
    #print(training_data_classifier.features)
    plt.scatter(X0, X1, cmap=plt.cm.coolwarm, s=1, edgecolors='k')
    plt.title("Classification boundary")
    plt.xlabel(training_data_classifier.headers[0])
    plt.ylabel(training_data_classifier.headers[1])
    plt.show()

def classifier_contour_3D(training_data_classifier,y_classifier, plot):
    fig = plot.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')


    points = []
    for i in range(ClassifierClient.num_model_in_classifier):

        points_i = ClassifierClient.get_cluster_points(i, y_classifier, training_data_classifier.features)
        points.append(points_i)
        #print(points_i)
        if points_i != []:
            ax.scatter(points_i[:,0], points_i[:,1], points_i[:,2], label=ClassifierClient.app_names_for_classifier[i])

    ax.set_xlabel(training_data_classifier.headers[0])
    ax.set_ylabel(training_data_classifier.headers[1])
    ax.set_zlabel(training_data_classifier.headers[2])
    ax.set_title("Classification boundary.")
    ax.legend()
    #ax.view_init(60, 100)



    return plot

def classifier_contour_2D(training_data_classifier,y_classifier, plot):
    fig = plot.figure(figsize=(9,7))
    ax = fig.add_subplot(111 ) #projection='3d')


    points = []
    for i in range(ClassifierClient.num_model_in_classifier):

        points_i = ClassifierClient.get_cluster_points(i, y_classifier, training_data_classifier.features)
        points.append(points_i)
        #print(points_i)
        if points_i != []:
            #print(i)
            #print(points_i)
            #print()
            ax.scatter(points_i[:,0], points_i[:,1], label=ClassifierClient.app_names_for_classifier[i])

    ax.set_xlabel(training_data_classifier.headers[0])
    ax.set_ylabel(training_data_classifier.headers[1])
    #ax.set_zlabel(training_data_classifier.headers[2])
    ax.set_title("Classification boundary.")
    ax.legend()



    return plot
