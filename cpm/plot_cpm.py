import Data as dt
import Client as client
import matplotlib.pyplot as plt
from matplotlib import gridspec
import vispy.plot as vp
import numpy as np
from vispy.color import ColorArray
from mpl_toolkits.mplot3d import axes3d






if __name__ == "__main__":
    # load the data
    # fields = ['n_tokens_title','n_tokens_content','n_non_stop_unique_tokens', 'n_unique_tokens']
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Temperature', 'energy_output']
    # fields = ['Exhaust_Vacuum','Relative_Humidity','energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/OnlineNewsPopularity1.csv",fields,y_column)
    data = dt.load_csv("datasets/6CCPP/Folds5x2_pp.csv", fields, y_column)
    global training_data_model
    training_data_model, training_data_classifier, testing_data = dt.split_data(data)

    training_data_model = training_data_model  # .get_before(500)
    training_data_classifier = training_data_classifier  # .get_before(500)
    testing_data = testing_data  # .get_before(500)

    '''
    fig = vp.Fig(show=False)
    color = (0.8, 0.25, 0.)
    fig1 = fig[0,0]
    fig1.plot(c,  symbol='o',width=0.0, marker_size=2.,color=r,face_color= g,edge_color=blue)
    '''
    # fig.show(run=True)

    models = client.deploy_all_models(training_data_model)

    answers_for_classifier = client.get_predictions_to_build_classifier(training_data_classifier)

    print(answers_for_classifier[0].NRMSE())
    print(answers_for_classifier[1].NRMSE())
    print(answers_for_classifier[2].NRMSE())
    print(answers_for_classifier[3].NRMSE())
    # print(answers_for_classifier[4].NRMSE())

    index = [0, 1, 2, 3]
    # index = [1,2]
    # index = None
    y_classifier, errors = client.init_classifier_training_values(answers_for_classifier, model_selection_index=index,
                                                                  factor=1)

    #####classifier = client.select_classifiers(training_data_classifier, y_classifier, testing_data)
    classifier = client.build_classifier_rbf(training_data_classifier, y_classifier, 100)

    predictions_classified = client.get_classified_predictions(classifier, testing_data)

    print(predictions_classified.NRMSE())

    answers_for_testing = client.get_predictions_from_models_for_testing(testing_data)
    print(answers_for_testing[0].NRMSE())
    print(answers_for_testing[1].NRMSE())
    print(answers_for_testing[2].NRMSE())
    print(answers_for_testing[3].NRMSE())
    # print(answers_for_testing[4].NRMSE())



    y_classifier, errors = client.init_classifier_training_values(answers_for_testing, model_selection_index=index,
                                                                  factor=1)
    a = np.array(testing_data.features)

    # b = np.array([np.array(training_data_model.labels)])
    # print(b)
    b = np.array(y_classifier).reshape(1, -1)

    # print(a)
    # print(b)
    c = np.concatenate((a, b.T), axis=1)

    #plot_classified_prediction_curves_2D(predictions_classified)


