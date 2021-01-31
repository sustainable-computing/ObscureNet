'''Keras code in someparts is borrowed from https://github.com/mmalekzadeh/motion-sense '''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
from keras.layers import Reshape, Lambda
import numpy as np
import pandas as pd
import dataset_builder as db
from keras.losses import mse, binary_crossentropy
import keras
import keras.backend as K
from collections import Counter
import matplotlib
matplotlib.use('Agg')
from keras import regularizers
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

ACT_LABELS = ["dws", "ups", "wlk", "jog", "std"]
TRIAL_CODES = {
    ACT_LABELS[0]: [1, 2, 11],
    ACT_LABELS[1]: [3, 4, 12],
    ACT_LABELS[2]: [7, 8, 15],
    ACT_LABELS[3]: [9, 16],
    ACT_LABELS[4]: [6, 14],
}

from keras.utils import to_categorical

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sampling_mean(z_mean, z_log_var):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    batch = z_mean.shape[0]
    dim = z_mean.shape[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = np.random.normal(size=(batch, dim))
    print(type(epsilon))
    return z_mean + np.exp(0.5 * z_log_var) * epsilon

class Estimator:
    l2p = 0.001
    @staticmethod
    def early_layers(inp, fm, hid_act_func="relu"):
        # Start
        x = Conv2D(64, fm, padding="same", kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(
            inp)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.25)(x)

        # 1
        x = Conv2D(64, fm, padding="same", kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(
            x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.25)(x)
        return x
    @staticmethod
    def late_layers(inp, num_classes, fm, act_func="softmax", hid_act_func="relu", b_name="Identifier"):
        # 2
        x = Conv2D(32, fm, padding="same", kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(
            inp)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.25)(x)

        # End
        x = Flatten()(x)
        x = Dense(128, kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(16, kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation=act_func, name=b_name)(x)

        return x

    @staticmethod
    def build(height, width, num_classes, name, fm, act_func="softmax", hid_act_func="relu"):
        inp = Input(shape=(height, width, 1))
        early = Estimator.early_layers(inp, fm, hid_act_func=hid_act_func)
        late = Estimator.late_layers(early, num_classes, fm, act_func=act_func, hid_act_func=hid_act_func)
        model = Model(inputs=inp, outputs=late, name=name)
        return model

class Estimator_mlp:
    l2p = 0.001
    @staticmethod
    def early_layers(inp, fm, hid_act_func="relu"):
        x = Dense(512, activation="relu")(inp)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        return x
    @staticmethod
    def late_layers(inp, num_classes, fm, act_func="softmax", hid_act_func="relu", b_name="Identifier"):
        # 2
        x = Dense(128, kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(inp)
        x = Dropout(0.5)(x)
        x = Dense(16, kernel_regularizer=regularizers.l2(Estimator.l2p), activation=hid_act_func)(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation=act_func, name=b_name)(x)
        return x
    @staticmethod
    def build(height, width, num_classes, name, fm, act_func="softmax", hid_act_func="relu"):
        inp = Input(shape=(256,))
        early = Estimator_mlp.early_layers(inp, fm, hid_act_func=hid_act_func)
        late = Estimator_mlp.late_layers(early, num_classes, fm, act_func=act_func, hid_act_func=hid_act_func)
        model = Model(inputs=inp, outputs=late, name=name)
        return model

def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

def eval_act(X, Y, act_class_numbers=5, fm=(2, 5), ep=200):
    height = 1
    width = 2
    ## Callbacks
    eval_metric = "val_acc"
    filepath = "activity_model"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=20)
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=0, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]
    eval_act = Estimator_mlp.build(height, width, act_class_numbers, name="EVAL_ACT", fm=fm, act_func="softmax",
                                   hid_act_func="relu")
    eval_act.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    # X, Y = shuffle(X, Y)
    eval_act.fit(X, Y,
                 validation_split=.1,
                 epochs=ep,
                 batch_size=128,
                 shuffle=True,
                 verbose=1,
                 class_weight=get_class_weights(np.argmax(Y, axis=1))
                 )

    eval_act.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print_results(eval_act, X, Y)
    eval_act.save("activity_model_mlp.hdf5")


def eval_gen(X, Y, gen_class_numbers=1, fm=(2, 5), ep=200, File="gender_model_mlp.hdf5"):
    height = 1
    width = 2
    ## Callbacks
    eval_metric = "val_acc"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=15)
    filepath = File
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=2, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]
    eval_gen = Estimator_mlp.build(height, width, gen_class_numbers, name="EVAL_GEN", fm=fm, act_func="sigmoid",
                                   hid_act_func="relu")
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    eval_gen.fit(X, Y,
                 epochs=ep,
                 batch_size=512,
                 verbose=1,
                 class_weight=get_class_weights(Y)
                 )
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    eval_gen.save(File)

#Used for training of Inference Models
def train_estimators(x, y, Gen=False):
    print("Training:")
    DS = DataSampler()
    train_data = DS.all_train()
    test_data = DS.all_test()
    gen_train_labels = DS.all_gender_train_labels()
    gen_test_labels = DS.all_gender_test_labels()
    act_train_labels = DS.all_train_labels()
    act_train = DS.get_act_train()
    act_test = DS.get_act_test()
    train_data_gender = train_data[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    gen_train_labels_gender = gen_train_labels[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    act_train_labels_gender = act_train_labels[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    if Gen == True:
        eval_gen(x, y, File="eval_gen_test.hdf5")
    else:
        eval_act(train_data, act_train_labels)

class DataSampler(object):
    def __init__(self):
        self.shape = [2, 128, 1]
        ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
        ## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
        self.sdt = ["rotationRate", "userAcceleration"]
        self.mode = "mag"
        self.cga = True  # Add gravity to acceleration or not
        print("[INFO] -- Selected sensor data types: " + str(self.sdt) + " -- Mode: " + str(
            self.mode) + " -- Grav+Acc: " + str(
            self.cga))

        self.act_labels = ACT_LABELS[0:5]
        print("[INFO] -- Selected activites: " + str(self.act_labels))
        self.trial_codes = [TRIAL_CODES[act] for act in self.act_labels]
        self.dt_list = db.set_data_types(self.sdt)
        self.dataset = db.creat_time_series(self.dt_list, self.act_labels, self.trial_codes, mode=self.mode,
                                            labeled=True, combine_grav_acc=self.cga)
        print("[INFO] -- Shape of time-Series dataset:" + str(self.dataset.shape))

        self.test_trail = [11, 12, 13, 14, 15, 16]
        print("[INFO] -- Test Trials: " + str(self.test_trail))
        self.test_ts = self.dataset.loc[(self.dataset['trial'].isin(self.test_trail))]
        self.train_ts = self.dataset.loc[~(self.dataset['trial'].isin(self.test_trail))]

        print("[INFO] -- Shape of Train Time-Series :" + str(self.train_ts.shape))
        print("[INFO] -- Shape of Test Time-Series :" + str(self.test_ts.shape))

        ## This Variable Defines the Size of Sliding Window
        ## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
        w = 128  # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
        ## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
        ## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
        s = 10
        self.train_data, self.act_train, self.id_train, self.train_mean, self.train_std = db.ts_to_secs(self.train_ts.copy(),
                                                                               w,
                                                                               s,
                                                                               standardize=True)

        s = 10
        self.test_data, self.act_test, self.id_test, self.test_mean, self.test_std = db.ts_to_secs(self.test_ts.copy(),
                                                                          w,
                                                                          s,
                                                                          standardize=True,
                                                                          mean=self.train_mean,
                                                                          std=self.train_std)

        # ## Here we add an extra dimension to the datasets just to be ready for using with Convolution2D
        # self.train_data = np.expand_dims(self.train_data, axis=3)
        # print("[INFO] -- Shape of Training Sections:", self.train_data.shape)
        # self.test_data = np.expand_dims(self.test_data, axis=3)
        # print("[INFO] -- Shape of Test Sections:", self.test_data.shape)

        self.size_train_data = self.train_data.shape[0]
        self.train_data = np.reshape(self.train_data, [self.size_train_data, 256])
        #
        self.size_test_data = self.test_data.shape[0]
        self.test_data = np.reshape(self.test_data, [self.size_test_data, 256])



        self.act_train_labels = to_categorical(self.act_train)
        self.act_test_labels = to_categorical(self.act_test)
        self.id_train_labels = to_categorical(self.id_train)
        self.id_test_labels = to_categorical(self.id_test)

        data_subject_info = pd.read_csv("data_subjects_info.csv")
        id_gen_info = data_subject_info[["code", "gender"]].values
        gen_id_dic = {item[0]: item[1] for item in id_gen_info}

        tmp = self.id_train.copy()
        gen_train = np.array([gen_id_dic[item + 1] for item in tmp])
        self.gen_train_labels = (gen_train).copy()

        tmp = self.id_test.copy()
        gen_test = np.array([gen_id_dic[item + 1] for item in tmp])
        self.gen_test_labels = (gen_test).copy()
    def next_batch(self, num, data, labels):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
        # return np.asarray(data), np.asarray(labels)
    def train(self, batch_size, label=False):
        if label:
            return self.next_batch(batch_size, self.train_data, self.train_labels)
        else:
            return self.next_batch(batch_size, self.train_data, self.train_labels)[0]
    def w_all_train(self):
        return self.w_train_data
    def w_all_train_gender(self):
        return self.w_gen_train_labels
    def w_all_test(self):
        return self.w_test_data
    def get_act_test(self):
        return self.act_test
    def get_act_train(self):
        return self.act_train
    def w_all_test_gender(self):
        return self.w_gen_test_labels
    def all_train(self):
        return self.train_data
    def all_train_labels(self):
        return self.act_train_labels
    def all_gender_train_labels(self):
        return self.gen_train_labels
    def all_gender_test_labels(self):
        return self.gen_test_labels
    def all_test(self):
        return self.test_data
    def all_test_labels(self):
        return self.act_test_labels

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(256,), name='encoder_input')
x = Dense(4096, activation='relu')(inputs)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation="relu")(x)
z_mean = Dense(10, name='z_mean')(x)
z_log_var = Dense(10, name='z_log_var')(x)
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(10,), name='z')([z_mean, z_log_var])
# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# build decoder model
latent_inputs = Input(shape=(10,), name='z_sampling')
x = Dense(128, activation="relu")(latent_inputs)
x = Dense(256, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
outputs = Dense(256, activation='linear')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
vae.summary()

DS = DataSampler()
x_train = DS.all_train()
y_train = DS.all_train_labels()
act_train_labels = DS.all_train_labels()
gen_train_labels = DS.all_gender_train_labels()
act_train = DS.get_act_train()

x_test = DS.all_test()
y_test = DS.all_test_labels()
act_test_labels = DS.all_test_labels()
gen_test_labels = DS.all_gender_test_labels()

x_train = x_train[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]

reconstruction_loss = mse(inputs, outputs)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.1
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# Training and fitting the VAE
# vae.fit(x_train, x_train,
#                 epochs=100,
#                 batch_size=128,
#                 validation_data=(x_test, x_test))
# vae.save_weights('mlp_vae.h5')

# Training the Inference networks for desired and intrusive inferences
# train_estimators(1, 2)

vae.load_weights('mlp_vae.h5')

Latent_means = np.zeros((5, 2, 10))

for activity_index in range(4):
    #Obtianing the Average values for the latent variables
    train_data = DS.all_train()
    act_train_labels = DS.all_train_labels()
    gen_train_labels = DS.all_gender_train_labels()
    act_train = DS.get_act_train()

    train_data = train_data[act_train_labels[:, activity_index]==1]
    gen_train_labels = gen_train_labels[act_train_labels[:, activity_index]==1]
    act_train_labels = act_train_labels[act_train_labels[:, activity_index]==1]
    act_train = act_train[act_train == activity_index]

    ### Manipulation at the Gender Level
    train_data_0 = train_data[gen_train_labels == 0]
    act_train_labels_0 = act_train_labels[gen_train_labels == 0]
    act_train_0 = act_train[gen_train_labels == 0]

    train_data_1 = train_data[gen_train_labels == 1]
    act_train_labels_1 = act_train_labels[gen_train_labels == 1]
    act_train_1 = act_train[gen_train_labels == 1]

    gender_train_data_0 = np.zeros((train_data_0.shape[0]))
    gender_train_data_1 = np.ones((train_data_1.shape[0]))

    eval_act_model = load_model("activity_model_mlp.hdf5")
    eval_gen_model = load_model("gender_model_mlp.hdf5")

    Latent_train_0 = encoder.predict(train_data_0)
    Latent_train_1 = encoder.predict(train_data_1)

    z_train_0 = Latent_train_0[2]
    z_train_1 = Latent_train_1[2]

    Latent_means[activity_index, 0, :] = np.mean(z_train_0, axis=0)
    Latent_means[activity_index, 1, :] = np.mean(z_train_1, axis=0)

#Labels used for F1-score calculation
ACT_LABELS = ["dws","ups", "wlk", "jog", "std"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
}
act_labels = ACT_LABELS [0:4]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
X_all = np.empty([0, x_test.shape[1]])
Y_all_act = np.empty([0, 5])
Y_all_gen = np.empty([0])
X_original = np.empty([0, x_test.shape[1]])

def print_act_results_f1_score(M, X, Y):
    result1 = M.evaluate(X, Y, verbose = 2)
    act_acc = round(result1[1], 4)*100
    print("***[RESULT]*** ACT Accuracy: "+str(act_acc))

    preds = M.predict(X)
    preds = np.argmax(preds, axis=1)
    conf_mat = confusion_matrix(np.argmax(Y, axis=1), preds)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("***[RESULT]*** ACT  Confusion Matrix")
    print(" | ".join(act_labels))
    print(np.array(conf_mat).round(3)*100)  

    f1act = f1_score(np.argmax(Y, axis=1), preds, average=None).mean()
    print("***[RESULT]*** ACT Averaged F-1 Score : "+str(f1act*100))

def print_gen_results_f1_score(M, X, Y):
    result1 = M.evaluate(X, Y, verbose = 2)
    act_acc = round(result1[1], 4)*100
    print("***[RESULT]*** Gender Accuracy: "+str(act_acc))

    preds = M.predict(X)
    preds_two_d = np.zeros((preds.shape[0], 2))
    for lop in range(preds.shape[0]):
        if preds[lop] < 0.5:
            preds_two_d[lop, 0] = 1
        else:
            preds_two_d[lop, 1] = 1
    Y_two_d = np.zeros((Y.shape[0], 2))
    for lop in range(Y.shape[0]):
        if Y[lop] == 0:
            Y_two_d[lop, 0] = 1
        else:
            Y_two_d[lop, 1] = 1
    preds_two_d = np.argmax(preds_two_d, axis=1)
    conf_mat = confusion_matrix(np.argmax(Y_two_d, axis=1), preds_two_d)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("***[RESULT]*** Gender  Confusion Matrix")
    print(" | ".join(act_labels))
    print(np.array(conf_mat).round(3)*100)  

    f1act = f1_score(np.argmax(Y_two_d, axis=1), preds_two_d, average=None).mean()
    print("***[RESULT]*** Gender Averaged F-1 Score : "+str(f1act*100))

train_data = DS.all_test()
act_train_labels = DS.all_test_labels()
gen_train_labels = DS.all_gender_test_labels()
act_train = DS.get_act_test()
train_data = train_data[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
gen_train_labels = gen_train_labels[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
act_train_labels = act_train_labels[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]

#Testing the General VAE method on test data
for activity_index in range(4):
    train_data = DS.all_test()
    act_train_labels = DS.all_test_labels()
    gen_train_labels = DS.all_gender_test_labels()
    act_train = DS.get_act_test()

    train_data = train_data[act_train_labels[:, activity_index]==1]
    gen_train_labels = gen_train_labels[act_train_labels[:, activity_index]==1]
    act_train_labels = act_train_labels[act_train_labels[:, activity_index]==1]
    act_train = act_train[act_train == activity_index]

    train_data = train_data[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    gen_train_labels = gen_train_labels[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    act_train_labels = act_train_labels[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]

    X = train_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)
    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: GEN Train Accuracy Gender 0: " + str(act_acc))
    
    data_mean_index = np.zeros((train_data.shape[0], 2, 1))

    X = train_data
    Y_act = eval_act_model.predict(X)

    for index in range(train_data.shape[0]):
        data_mean_index[index, 0] = np.argmax(Y_act[index], axis=0)

    Y_gen = eval_gen_model.predict(X)

    for index in range(train_data.shape[0]):
        if Y_gen[index] > 0.5:
            data_mean_index[index, 1] = 1
        else:
            data_mean_index[index, 1] = 0

    Latent_reps = encoder.predict(train_data)[2]

    for index in range(train_data.shape[0]):
        act = int(data_mean_index[index, 0, 0])
        mean_0 = Latent_means[act, 0, :]
        mean_1 = Latent_means[act, 1, :]
        if data_mean_index[index, 1] == 0:
            Latent_reps[index] = Latent_reps[index] - mean_0 + mean_1
        else:
            Latent_reps[index] = Latent_reps[index] - mean_1 + mean_0

    hat_train_data = decoder.predict(Latent_reps)
    if activity_index == 0:
        reconstructed_input = hat_train_data
        act_label = act_train_labels
        gen_label = gen_train_labels
    else:
        reconstructed_input = np.concatenate((reconstructed_input, hat_train_data), axis=0)
        act_label = np.concatenate((act_label, act_train_labels), axis=0)
        gen_label = np.concatenate((gen_label, gen_train_labels), axis=0)

    X = hat_train_data
    X_all = np.append(X_all, X, axis=0)

    Y = act_train_labels
    Y_all_act = np.append(Y_all_act, Y, axis=0)

    Y = gen_train_labels
    Y_all_gen = np.append(Y_all_gen, Y, axis=0)

print_act_results_f1_score(eval_act_model, X_all, Y_all_act)
print_gen_results_f1_score(eval_gen_model, X_all, Y_all_gen)
