from sklearn.cluster import KMeans
import  Data as dt
import ClientClass

class Cluster_Kmeans:
    def __init__(self):
        self.num_cluster=10
        self.ccs=[]
        self.kmeans=None

    def fit(self,training_data,testing_data,num_cluster=10):


        headers = training_data.headers
        features_training_data = training_data.features
        labels_training_data = training_data.labels

        features_testing_data = testing_data.features
        labels_testing_data = testing_data.labels

        self.kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(features_training_data)
        self.num_cluster=num_cluster

        self.ccs = []
        for i in range(self.num_cluster):
            index = self.kmeans.labels_ == i
            ds_training_data = dt.DataSource()
            ds_training_data.features = features_training_data[index, :]
            ds_training_data.labels = labels_training_data[index]
            ds_training_data.headers = headers

            ds_training_data_model, ds_training_data_classifier = dt.split_data_to_2(ds_training_data, 0.5)

            # ds_training_data_classifier.features = features_training_data_classifier[index, :]
            # ds_training_data_classifier.labels = labels_training_data_classifier[index]
            # ds_training_data_classifier.headers=headers

            cc = ClientClass.ClientClass()
            # cc.fit(ds_training_data_model,ds_training_data_model)
            # print(cc.predict(testing_data.features[0]))
            print("train the "+str(i)+" cluster. ---------------------------------------------------------------------------------------"+"train the "+str(i)+" cluster.")
            cc.fit(ds_training_data_model, ds_training_data_classifier,testing_data,b_select_classifier=False)

            self.ccs.append(cc)

        return

    def predict(self,x):
        cluster_id=self.kmeans.predict([x])
        #print("point is: "+str(x)+", the cluster id is: "+str(cluster_id))
        return self.ccs[cluster_id].predict(x)

    def predicts(self,xs):
        return [self.predict(x) for x in xs]




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
    #'''
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
    #'''

    # Number 3 dataset
    '''
    fields = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    fields = ['pm2.5','PRES']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    fields = ['pm2.5','TEMP', 'PRES']
    fields = ['pm2.5', 'DEWP', 'TEMP']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/3PRSA_data.csv", fields, y_column)
    '''

    # Number 4 dataset
    '''
    # load the data
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    fields = [ 'n_tokens_content', 'n_unique_tokens']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    fields = [ 'n_tokens_content','n_unique_tokens','n_non_stop_unique_tokens']
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens']
    y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/OnlineNewsPopularity1.csv", fields, y_column)
    '''

    # Number 5 dataset
    '''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Exhaust_Vacuum', 'energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    fields = ['Exhaust_Vacuum', 'Ambient_Pressure', 'energy_output']
    fields = ['Temperature', 'Ambient_Pressure', 'energy_output']
    y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/6CCPP/Folds5x2_pp.csv", fields, y_column)
    '''


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
    fields = ['year', 'c1','c2']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
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
    fields = ['c1', 'c2','c4']
    fields = ['Methane_conc_(ppm)','c1', 'c2',]
    y_column = 2  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/7/ethylene_methane_with_header.csv", fields, y_column, sep=' ')
    '''

    training_data, testing_data = dt.split_data_to_2(data, percent=0.70)
    ck=Cluster_Kmeans()
    ck.fit(training_data,testing_data,num_cluster=10)





    print(dt.NRMSE(ck.predicts(testing_data.features),testing_data.labels))

    #for model in ck.ccs:
    #    print(model.predict(testing_data.features[0]))








