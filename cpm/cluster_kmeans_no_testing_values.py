from sklearn.cluster import KMeans
import  Data as dt
import ClientClass

class Cluster_Kmeans_no_testing:
    """
    cluster the dataset, then select the single best model locally.
    Proved to be bad, not used.
    """
    def __init__(self):
        self.num_cluster=10
        self.local_models=[] # store the local best model in each cluster
        self.kmeans=None
        self.predictions_models_in_cluster=[]
        self.NRMSE_models_in_cluster=[]
        self.indexs_best_model=[]
        self.headers=None
        self.data=[]    # store the datasource of each cluster.

    def fit(self,training_data,testing_data,num_cluster=10):
        self.headers = training_data.headers
        #features_training_data = training_data.features
        #labels_training_data = training_data.labels

        #features_testing_data = testing_data.features
        #labels_testing_data = testing_data.labels

        #training_data_model, training_data_cluster = dt.split_data_to_2(training_data, 0.5)

        clientClass=ClientClass.ClientClass()
        models, time_cost_training=clientClass.deploy_all_models(training_data)



        self.kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(training_data.features)
        self.num_cluster=num_cluster


        for i in range(self.num_cluster):
            index = self.kmeans.labels_ == i
            predictions_i = list()
            NRMSE_i=list()
            ds_training_data = dt.DataSource()
            ds_training_data.features = training_data.features[index, :]
            ds_training_data.labels = training_data.labels[index]
            ds_training_data.headers = self.headers

            for index_model in range(len(models)):
                ys= models[index_model].predict(ds_training_data.features)
                predictions_i.append(ys)
                NRMSE_i.append(dt.NRMSE(ys,ds_training_data.labels))
            self.predictions_models_in_cluster.append(predictions_i)
            self.NRMSE_models_in_cluster.append(NRMSE_i)

            index_best_model = NRMSE_i.index(min(NRMSE_i))
            self.indexs_best_model.append(index_best_model)
            self.local_models.append(models[index_best_model])
            self.data.append(ds_training_data)
        #print(self.NRMSE_models_in_cluster)
        #print(self.indexs_best_model)
        #print(self.local_models)

        return


    def fit_local(self,training_data,testing_data,num_cluster=10):
        self.headers = training_data.headers

        #clientClass=ClientClass.ClientClass()
        #models_global, time_cost_training=clientClass.deploy_all_models(training_data)



        self.kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(training_data.features)
        self.num_cluster=num_cluster


        for i in range(self.num_cluster):
            index = self.kmeans.labels_ == i
            #print(index)
            predictions_i = list()
            NRMSE_i=list()
            ds_training_data = dt.DataSource()
            ds_training_data.features = training_data.features[index, :]
            ds_training_data.labels = training_data.labels[index]
            print(training_data.labels)
            print(index)
            ds_training_data.headers = self.headers

            clientClass_i = ClientClass.ClientClass()
            models,timecost=clientClass_i.deploy_all_models(ds_training_data)

            for index_model in range(4):
                ys= models[index_model].predict(ds_training_data.features)
                predictions_i.append(ys)
                NRMSE_i.append(dt.NRMSE(ys,ds_training_data.labels))
            self.predictions_models_in_cluster.append(predictions_i)
            self.NRMSE_models_in_cluster.append(NRMSE_i)

            index_best_model = NRMSE_i.index(min(NRMSE_i))
            self.indexs_best_model.append(index_best_model)
            self.local_models.append(models[index_best_model])
            self.data.append(ds_training_data)
        #print(self.NRMSE_models_in_cluster)
        #print(self.indexs_best_model)
        #print(self.local_models)

        return

    def predict(self,x):
        cluster_id=self.kmeans.predict([x])
        #print("point is: "+str(x)+", the cluster id is: "+str(cluster_id))
        return self.local_models[cluster_id].predict([x])

    def predicts(self,xs):
        return [self.predict(x) for x in xs]

    def plot1d_clusters(self):
        import matplotlib.pyplot as plt
        for i in range(self.num_cluster):
            self.data[i].sort1d()


            plt.plot(self.data[i].features,self.data[i].labels,
                        marker=dt.markers_matplotlib[i],

                        label="cluster "+str(i),
                        linewidth=0.0,
                        color=dt.colors_matploblib[i])
            plt.xlabel(self.headers[0])
            plt.ylabel(self.headers[1])
            plt.legend()

        plt.show()








if __name__=="__main__":
    # Number 1 dataset
    '''
    fields = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size',
              'b_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem', 'utime']
    fields = ['umem','utime']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    #fields = ['bitrate','framerate', 'utime']
    #y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/1online_video_dataset/1transcoding_mesurment.csv", fields, y_column)
    '''

    # Number 2 dataset
    '''
    # fields = ['Temperature','Exhaust_Vacuum','Relative_Humidity','energy_output']
    # fields = ['Temperature','Exhaust_Vacuum','energy_output']
    fields = ["RMSD", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    #fields = ["RMSD", "F2", "F3"]
    #fields = ["RMSD", "F2", "F7"]
    fields = ["RMSD", "F4", "F5"]
    fields = ["RMSD", "F1", "F2"]
    fields = ["RMSD", "F1"]
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/2CASP.csv", fields, y_column)
    '''

    # Number 3 dataset
    '''
    fields = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    fields = ['pm2.5','PRES']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    #fields = ['pm2.5','TEMP', 'PRES']
    #fields = ['pm2.5', 'DEWP', 'TEMP']
    #y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/3PRSA_data.csv", fields, y_column)
    '''

    # Number 4 dataset
    '''
    # load the data
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    fields = [ 'n_tokens_content', 'n_unique_tokens']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    #fields = [ 'n_tokens_content','n_unique_tokens','n_non_stop_unique_tokens']
    #fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens']
    #y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/OnlineNewsPopularity1.csv", fields, y_column)
    '''

    # Number 5 dataset
    #'''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Exhaust_Vacuum', 'energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    #fields = ['Temperature',  'energy_output']
    #fields = ['Temperature', 'Ambient_Pressure', 'energy_output']
    #y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/6CCPP/Folds5x2_pp.csv", fields, y_column)
    #'''


    # Number 6 dataset
    '''
    fields = ['year', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
              'c16', 'c17', 'c18', 'c19', 'c20',
              'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35',
              'c36', 'c37', 'c38', 'c39',
              'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54',
              'c55', 'c56', 'c57', 'c58',
              'c59', 'c60', 'c61', 'c62', 'c63', 'c64', 'c65', 'c66', 'c67', 'c68', 'c69', 'c70', 'c71', 'c72', 'c73',
              'c74', 'c75', 'c76', 'c77',
              'c78', 'c79', 'c80', 'c81', 'c82', 'c83', 'c84', 'c85', 'c86', 'c87', 'c88', 'c89', 'c90']
    fields = ['year', 'c1']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    #fields = ['year', 'c1','c2']
    #y_column = 0  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/6YearPredictionMSD_with_header.csv", fields, y_column)
    '''

    # Number 7 dataset
    '''
    # load the data
    # fields = ['duration','width','height','bitrate','framerate','i','p','b','frames','i_size','p_size','b_size','size','o_bitrate','o_framerate','o_width','o_height','umem','utime']

    fields = ['Time_(seconds)', 'Methane_conc_(ppm)', 'Ethylene_conc_(ppm)', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
              'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16']
    fields = ['c1', 'c2']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    #fields = ['c1', 'c2','c4']
    #fields = ['Methane_conc_(ppm)','c1', 'c2',]
    #y_column = 2  # should be the order in the input file, not in the "fields" order.

    data = dt.load_csv("../data/7/ethylene_methane_with_header.csv", fields, y_column, sep=' ')
    '''

    training_data, testing_data = dt.split_data_to_2(data, percent=0.70)
    ck=Cluster_Kmeans_no_testing()
    ck.fit_local(training_data,testing_data,num_cluster=10)

    ck.plot1d_clusters()


    print(ck.indexs_best_model)
    print(ck.NRMSE_models_in_cluster)
    print(dt.NRMSE(ck.predicts(testing_data.features),testing_data.labels))

    #for model in ck.ccs:
    #    print(model.predict(testing_data.features[0]))








