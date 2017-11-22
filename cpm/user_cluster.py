from sklearn.cluster import KMeans
import Data as dt
import ClientClass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


class UserCluster:
    """
    This class is used as the alternative method, use Gaussian or Zipffian distribution to generate the queries.
    """

    def __init__(self):
        self.num_cluster = 10
        self.local_models = []  # store the local best model in each cluster
        self.predictions_models_in_cluster = []
        self.NRMSE_models_in_cluster = []
        self.headers = None
        self.data = []  # store the datasource of each cluster.
        self.kmeans = None
        self.indexs_best_model=[]
        self.cluster_centroids=[]
        self.num_model_for_regression=None
        self.NRMSE_best=[]

    def fit(self, training_data, testing_data, num_cluster=3):
        self.headers = training_data.headers

        clientClass = ClientClass.ClientClass()
        models, time_cost_training = clientClass.deploy_all_models(training_data)

        queries = dt.mix_gaussian(10000,k=num_cluster, b_show_hist=False)
        #queries = dt.scaleBacks(queries, min(training_data.features), max(training_data.features))

        # plt.hist(queries,100)
        # plt.show()

        self.kmeans = KMeans(n_clusters=num_cluster, random_state=0,n_jobs=-1).fit(queries)
        self.num_cluster = num_cluster
        self.cluster_centroids=self.kmeans.cluster_centers_
        #print("centroids")
        #print(self.cluster_centroids)

        ys_query = self.get_neighbors_ys(queries, training_data, n_neighbors=5)

        #self.plot1d_clusters()



        for i in range(self.num_cluster):
            index = self.kmeans.labels_ == i
            predictions_i = list()
            NRMSE_i = list()
            ds_training_data = dt.DataSource()
            ds_training_data.features = queries[index, :]
            #print(index)
            #print(ys_query)
            ds_training_data.labels = np.asarray(ys_query)[index]
            ds_training_data.headers = self.headers

            for index_model in range(len(models)):
                ys = models[index_model].predict(ds_training_data.features)
                predictions_i.append(ys)
                NRMSE_i.append(dt.NRMSE(ys, ds_training_data.labels))
            self.predictions_models_in_cluster.append(predictions_i)
            self.NRMSE_models_in_cluster.append(NRMSE_i)

            index_best_model = NRMSE_i.index(min(NRMSE_i))
            self.NRMSE_best.append(min(NRMSE_i))

            self.local_models.append(models[index_best_model])
            self.data.append(ds_training_data)
            self.indexs_best_model.append(index_best_model)
        #print(self.NRMSE_models_in_cluster)
        #print(self.indexs_best_model)
        # print(self.local_models)

        return

    def predict(self, x,n_models_for_regression=2,b_combining_models=True):
        if b_combining_models:
            self.num_model_for_regression = n_models_for_regression
            cluster_id = self.kmeans.predict([x])
            distances = [self.distance(x, centre) for centre in self.cluster_centroids]
            # print(distances)
            distances, index = self.sortDistance(distances)
            distances = distances[0:self.num_model_for_regression]
            index = index[0:self.num_model_for_regression]
            NRMSEs = [self.NRMSE_best[i] for i in index]

            # print(distances)
            # print(index)
            # print(NRMSEs)
            # print("point is: "+str(x)+", the cluster id is: "+str(cluster_id))
            weights = self.weightFn(distances, NRMSEs)
            # print(weights)


            predictions = [self.local_models[ii].predict([x])[0] for ii in index]
            # print(predictions)
            return sum(w * p for w, p in zip(weights, predictions))
        else:
            cluster_id = self.kmeans.predict([x])
            # print("point is: "+str(x)+", the cluster id is: "+str(cluster_id))
            return self.local_models[cluster_id].predict([x])[0]


    def predicts(self, xs,b_combining_models=True):
        return [self.predict(x,b_combining_models=b_combining_models) for x in xs]

    def plot1d_clusters(self):
        import matplotlib.pyplot as plt
        for i in range(self.num_cluster):
            self.data[i].sort1d()

            plt.plot(self.data[i].features, self.data[i].labels,
                     marker=dt.markers_matplotlib[i],

                     label="cluster " + str(i),
                     linewidth=0.0,
                     color=dt.colors_matploblib[i])
            plt.xlabel(self.headers[0])
            plt.ylabel(self.headers[1])
            plt.legend()

        plt.show()

    def get_neighbors_ys(self, xs, trainingData, n_neighbors=1):

        tree = KDTree(trainingData.features)
        dis, ind = tree.query(xs, k=n_neighbors)
        #print(ind)
        if n_neighbors == 1:
            return trainingData.labels[ind]
        else:
            return [sum(i) / len(i) for i in trainingData.labels[ind]]

    def distance(self,a,b,type='euclidean'):
        return np.sqrt(np.sum(a-b)**2)

    def sortDistance(self,distances):
        zipped = zip(distances, range(len(distances)))
        zipped.sort()

        distances = [i for (i, j) in zipped]
        indexs = [j for (i, j) in zipped]
        return distances,indexs

    def weightFn(self,distances,NRMSEs):

        """
        This function is used to calculate the weight of prediction models from each cluster.

        Parameters
        ----------
        distances : list
            the distance from point to clusters
        NRMSEs : list
            the prediction NRMSE of local best model.
        Returns
        -------
        output : list
            the weight for point x to be predicted.
        """
        def costFn(distances,NRMSEs):
            """
            cost function, combining both distances and NRMSEs
            """
            return distances

        combinings=costFn(distances,NRMSEs)

        total=np.sum(np.array(combinings))

        return [(total-element)/total for element in combinings]



if __name__ == "__main__":
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
    fields = ['n_tokens_content', 'n_unique_tokens']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    # fields = [ 'n_tokens_content','n_unique_tokens','n_non_stop_unique_tokens']
    # fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens']
    # y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/OnlineNewsPopularity1.csv", fields, y_column)
    '''

    # Number 5 dataset
    #'''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Temperature', 'energy_output']
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

    # Number 8 dataset
    '''
    # load the data

    fields = ['timestamp', 'energy']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    #fields = ['c1', 'c2','c4']
    #fields = ['Methane_conc_(ppm)','c1', 'c2',]
    #y_column = 2  # should be the order in the input file, not in the "fields" order.

    data = dt.load_csv("../data/8data.txt", fields, y_column, sep=',')
    data=data.get_before(300000)

    '''
    data.remove_repeated_x_1d()

    training_data, testing_data = dt.split_data_to_2(data, percent=0.70)


    plt.plot(training_data.features,training_data.labels,linewidth=0.0,marker='*')
    plt.show()


    training_data.scale()


    ck = UserCluster()
    ck.fit(training_data, testing_data,num_cluster=3)

    #print(ck.predict(training_data.features[0]))

    # ck.plot1d_clusters()
    # plt.plot(training_data.features,training_data.labels,marker='x',linewidth=0.0)
    # plt.show()

    #print(ck.NRMSE_models_in_cluster)
    print(ck.indexs_best_model)

    queries = dt.mix_gaussian(1000, k=ck.num_cluster, b_show_hist=False)

    ys_query = ck.get_neighbors_ys(queries, training_data, n_neighbors=1)

    print(dt.NRMSE(ck.predicts(queries,b_combining_models=True),ys_query))
    print(dt.NRMSE(ck.predicts(queries, b_combining_models=False), ys_query))



