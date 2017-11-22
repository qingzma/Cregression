from __future__ import print_function
import Data as dt
from ClientClass import ClientClass
import os
import logging.config
import json
import logging

logger_name = __file__.split(os.path.sep)[-2]
logger = logging.getLogger(logger_name)


def setup_logging(
        default_path='logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration using json file.

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def set_logging():
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('result.log', mode='w')
    fh.setLevel(logging.ERROR)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def run2d1(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 1 dataset

    fields = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size',
              'b_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem', 'utime']
    fields = ['utime', 'umem']
    fields = ['i_size', 'umem']
    # fields = ['bitrate', 'framerate']
    # fields = ['umem', 'utime']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    # fields = ['bitrate','framerate', 'utime']
    # y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/1online_video_dataset/1transcoding_mesurment.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d2(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 2 dataset

    fields = ["RMSD", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    fields = ["RMSD", "F2"]
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/2CASP.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d3(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 3 dataset

    fields = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    fields = ['pm2.5', 'Iws']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/3PRSA_data.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d4(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 4 dataset

    # load the data
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    fields = ['n_tokens_content', 'n_unique_tokens']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    # fields = ['n_unique_tokens', 'n_non_stop_unique_tokens']
    # y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/4OnlineNewsPopularity1.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d5(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 5 dataset
    # '''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Relative_Humidity', 'energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/5CCPP/5Folds5x2_pp.csv", fields, y_column)
    # '''

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d6(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 6 dataset

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
    data = dt.load_csv("../data/6YearPredictionMSD_with_header.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d7(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 7 dataset

    # load the data
    # fields = ['duration','width','height','bitrate','framerate','i','p','b','frames','i_size','p_size','b_size','size','o_bitrate','o_framerate','o_width','o_height','umem','utime']

    fields = ['Time_(seconds)', 'Methane_conc_(ppm)', 'Ethylene_conc_(ppm)', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
              'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16']
    fields = ['c1', 'c2']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/7/7ethylene_methane_with_header.csv", fields, y_column, sep=' ')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


def run2d8(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 8 dataset

    # Number 8 dataset

    # load the data

    fields = ['timestamp', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
              'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'energy']
    # fields = ['Date',  'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    #          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    fields = ['timestamp', 'energy']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/8data.txt", fields, y_column, sep=',')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run2d(data)


#########################################################################################################################################################################################
def run3d1(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 1 dataset

    fields = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size',
              'b_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem', 'utime']
    fields = ['i_size', 'umem', 'utime']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/1online_video_dataset/1transcoding_mesurment.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d2(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 2 dataset
    fields = ["RMSD", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    fields = ["RMSD", "F3", "F5"]

    # fields = ["RMSD", "F2", "F7"]
    fields = ["RMSD", "F4", "F5"]

    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/2CASP.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d3(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 3 dataset

    fields = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    fields = ['pm2.5', 'PRES']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    fields = ['pm2.5', 'TEMP', 'PRES']
    fields = ['pm2.5', 'TEMP', 'Iws']  # good vision
    fields = ['pm2.5', 'PRES', 'Iws']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/3PRSA_data.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d4(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 4 dataset

    # load the data
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    fields = ['n_tokens_content', 'n_unique_tokens']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens']
    fields = ['n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    # fields = [ 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/4OnlineNewsPopularity1.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d5(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 5 dataset
    # '''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    fields = ['Exhaust_Vacuum', 'energy_output']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    # fields = ['Exhaust_Vacuum', 'Ambient_Pressure', 'energy_output']
    fields = ['Temperature', 'Ambient_Pressure', 'energy_output']
    fields = ['Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    # fields = ['Exhaust_Vacuum', 'Ambient_Pressure', 'energy_output']
    # fields = ['Temperature', 'Exhaust_Vacuum', 'energy_output']


    y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/5CCPP/5Folds5x2_pp.csv", fields, y_column)
    # '''
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d6(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 6 dataset

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
    fields = ['year', 'c1', 'c2']
    # fields = ['year', 'c2', 'c4']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/6YearPredictionMSD_with_header.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d7(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 7 dataset

    # load the data
    # fields = ['duration','width','height','bitrate','framerate','i','p','b','frames','i_size','p_size','b_size','size','o_bitrate','o_framerate','o_width','o_height','umem','utime']

    fields = ['Time_(seconds)', 'Methane_conc_(ppm)', 'Ethylene_conc_(ppm)', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
              'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16']
    fields = ['c1', 'c2']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    fields = ['c1', 'c2', 'c4']
    fields = ['Methane_conc_(ppm)', 'c1', 'c2']
    y_column = 2  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/7/7ethylene_methane_with_header.csv", fields, y_column, sep=' ')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


def run3d8(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 8 dataset

    # load the data

    fields = ['timestamp', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
              'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'energy']
    # fields = ['Date',  'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    #          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    fields = ['Global_active_power', 'Global_reactive_power', 'energy']
    y_column = 2  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/8data.txt", fields, y_column, sep=',')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run3d(data)


# ----------------------------------------------------------------------------------------------------------------------#

#########################################################################################################################################################################################
def run1(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 1 dataset

    fields = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size',
              'b_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem', 'utime']
    # fields = ['i_size', 'umem','utime']
    fields = ['duration', 'bitrate', 'framerate', 'size', 'umem', 'utime']
    y_column = 4  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/1online_video_dataset/1transcoding_mesurment.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run2(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 2 dataset
    fields = ["RMSD", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    # fields = ["RMSD", "F3", "F5"]

    # fields = ["RMSD", "F2", "F7"]
    fields = ["RMSD", "F4", "F5"]

    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/2CASP.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run3(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 3 dataset

    fields = ['year', 'month', 'day', 'hour', 'pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    y_column = 4
    # fields = ['pm2.5','PRES']
    # y_column = 0  # should be the order in the input file, not in the "fields" order.
    # fields = ['pm2.5','TEMP', 'PRES']
    # fields = ['pm2.5', 'TEMP', 'Iws'] # good vision
    # fields = ['pm2.5', 'PRES', 'Iws']
    # y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/3PRSA_data.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run4(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 4 dataset

    # load the data
    fields = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    y_column = 2
    # fields = ['n_tokens_content', 'n_unique_tokens']
    # y_column = 1  # should be the order in the input file, not in the "fields" order.
    # fields = ['n_tokens_title','n_tokens_content','n_unique_tokens']
    # fields = ['n_tokens_content', 'n_unique_tokens','n_non_stop_unique_tokens']
    # #fields = [ 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens']
    # y_column = 1  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/4OnlineNewsPopularity1.csv", fields, y_column)
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run5(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 5 dataset
    # '''
    # load the data
    fields = ['Temperature', 'Exhaust_Vacuum', 'Ambient_Pressure', 'Relative_Humidity', 'energy_output']
    y_column = 4  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/5CCPP/5Folds5x2_pp.csv", fields, y_column)
    # '''
    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run6(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 6 dataset

    fields = ['year', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
              'c16', 'c17', 'c18', 'c19', 'c20',
              'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35',
              'c36', 'c37', 'c38', 'c39',
              'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54',
              'c55', 'c56', 'c57', 'c58',
              'c59', 'c60', 'c61', 'c62', 'c63', 'c64', 'c65', 'c66', 'c67', 'c68', 'c69', 'c70', 'c71', 'c72', 'c73',
              'c74', 'c75', 'c76', 'c77',
              'c78', 'c79', 'c80', 'c81', 'c82', 'c83', 'c84', 'c85', 'c86', 'c87', 'c88', 'c89', 'c90']
    fields = ['year', 'c1', 'c2', 'c3']
    y_column = 0  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/6YearPredictionMSD_with_header.csv", fields, y_column)

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run7(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 7 dataset

    # load the data
    # fields = ['duration','width','height','bitrate','framerate','i','p','b','frames','i_size','p_size','b_size','size','o_bitrate','o_framerate','o_width','o_height','umem','utime']

    fields = ['Time_(seconds)', 'Methane_conc_(ppm)', 'Ethylene_conc_(ppm)', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
              'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16']
    fields = ['c1', 'c2']
    y_column = 1  # should be the order in the input file, not in the "fields" order.
    fields = ['c1', 'c2', 'c4']
    fields = ['Methane_conc_(ppm)', 'Ethylene_conc_(ppm)', 'c1', 'c2', 'c3', 'c4']
    y_column = 3  # should be the order in the input file, not in the "fields" order.
    # data = dt.load_csv("datasets/1online_video_dataset/1transcoding_mesurment.csv",fields,y_column)
    data = dt.load_csv("../data/7/7ethylene_methane_with_header.csv", fields, y_column, sep=' ')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)


def run8(base_models=None, ensemble_models=None,classifier_type='xgboost', b_show_plot=False):
    # Number 8 dataset

    # load the data

    fields = ['timestamp', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
              'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'energy']
    # fields = ['Date',  'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    #          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    # fields = ['Global_active_power', 'Global_reactive_power', 'energy']
    y_column = 8  # should be the order in the input file, not in the "fields" order.
    data = dt.load_csv("../data/8data.txt", fields, y_column, sep=',')

    client = ClientClass(logger_name=logger_name, base_models=base_models, ensemble_models=ensemble_models,
                         classifier_type=classifier_type,b_show_plot=b_show_plot)

    client.run_all_features(data)



if __name__ == "__main__":
    base_model_library = ["sklearn_linear", "sklearn_poly", "sklearn_decision_tree", "sklearn_knn",
                          "sklearn_svr_rbf", "mllib_regression", "sklearn_gaussian_process",
                          "sklearn_adaboost", "sklearn_gradient_tree_boosting", "xgboost"]
    ensemble_model_library = ["sklearn_adaboost", "sklearn_gradient_tree_boosting", "xgboost"]
    # setup_logging()
    set_logging()
    # client =ClientClass(logger_name)
    # client.test_logger()
    # run2d4(base_models=["sklearn_linear", "sklearn_poly"],
    #        ensemble_models=["sklearn_gradient_tree_boosting", "xgboost"])
    run2d5(base_models=["sklearn_linear", "sklearn_poly"],
           ensemble_models=["sklearn_gradient_tree_boosting", "xgboost"],
           classifier_type='xgboost',b_show_plot=True)
