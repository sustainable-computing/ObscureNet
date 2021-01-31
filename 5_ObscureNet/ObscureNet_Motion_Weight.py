'''Keras code in someparts is borrowed from https://github.com/mmalekzadeh/motion-sense '''

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
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
from keras.utils import to_categorical

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
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

def eval_act(X, Y, act_class_numbers=5, fm=(2, 5), ep=10):
    height = X.shape[1]
    width = X.shape[2]
    ## Callbacks
    eval_metric = "val_acc"
    filepath = "activity_model"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=20)
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=0, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]

    eval_act = Estimator.build(height, width, act_class_numbers, name="EVAL_ACT", fm=fm, act_func="softmax",
                                   hid_act_func="relu")
    eval_act.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    eval_act.fit(X, Y,
                 validation_split=.1,
                 epochs=ep,
                 batch_size=128,
                 shuffle=True,
                 verbose=2,
                 class_weight=get_class_weights(np.argmax(Y, axis=1))
                 )

    eval_act.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print_results(eval_act, X, Y)
    eval_act.save("activity_model_DC.hdf5")

def eval_gen(X, Y, gen_class_numbers=1, fm=(2, 5), ep=10, File="gender_model_DC.hdf5"):
    height = X.shape[1]
    width = X.shape[2]
    ## Callbacks
    eval_metric = "val_acc"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=15)
    filepath = File
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=2, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]

    eval_gen = Estimator.build(height, width, gen_class_numbers, name="EVAL_GEN", fm=fm, act_func="sigmoid",
                                   hid_act_func="relu")
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    eval_gen.fit(X, Y,
                 epochs=ep,
                 batch_size=512,
                 verbose=2,
                 class_weight=get_class_weights(Y)
                 )
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    eval_gen.save(File)


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

def train_estimators():
    print("Training:")
    DS = DataSampler()
    train_data = DS.all_train()
    test_data = DS.all_test()
    gen_train_labels = DS.all_gender_train_labels()
    gen_test_labels = DS.all_gender_test_labels()
    act_train_labels = DS.all_train_labels()
    act_train = DS.get_act_train()
    act_test = DS.get_act_test()
    train_data_all = train_data[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    gen_train_labels_all = gen_train_labels[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    act_train_labels_all = act_train_labels[
        np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]
    train_data_all = np.reshape(train_data_all, (train_data_all.shape[0], 2, 128, 1))
    eval_gen(train_data_all, gen_train_labels_all)
    eval_act(train_data_all, act_train_labels_all)

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
from keras.layers import Reshape, Lambda
import numpy as np
import pandas as pd
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
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.nn import functional as F

#Pytorch models
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import math, os

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

usecuda = True
use_gpu = True
idgpu = 0
x_dim = 2 #Dimension of added Condition Variable
zed = [10] #Latent variable size

for z_dim in zed:
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(256, 512)
            # self.fc2 = nn.Linear(512,512)
            self.fc3 = nn.Linear(512,256)
            self.fc4 = nn.Linear(256,128)
            self.z_mean = nn.Linear(128, z_dim)
            self.z_log_var = nn.Linear(128, z_dim)
            self.relu = nn.ReLU()

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self,x):
            h1 = self.relu(self.fc1(x))
            # h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h1))
            h4 = self.relu(self.fc4(h3))
            z_m = self.z_mean(h4)
            z_l = self.z_log_var(h4)
            z = self.reparameterize(z_m, z_l)
            return z, z_m, z_l

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(z_dim+x_dim, 128)
            self.fc2 = nn.Linear(128,256)
            self.fc3 = nn.Linear(256,512)
            # self.fc4 = nn.Linear(512,512)
            self.fc5 = nn.Linear(512, 256)
            self.relu = nn.ReLU()

        def forward(self,x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            # h4 = self.relu(self.fc4(h3))
            h5 = self.fc5(h3)
            return h5
    
    class AUX(nn.Module):
        def __init__(self, nz, numLabels=2):
            super(AUX, self).__init__()
            self.nz = nz
            self.numLabels = numLabels
            self.aux1 = nn.Linear(nz, 128)
            self.aux3 = nn.Linear(128, numLabels)
        def infer_y_from_z(self, z):
            z = F.relu(self.aux1(z))
            if self.numLabels==1:
                z = F.sigmoid(self.aux3(z))
            else:
                z = F.softmax(self.aux3(z))
            return z
        def forward(self, z):
            return self.infer_y_from_z(z)
        def loss(self, pred, target):
            return F.nll_loss(pred, target)
    
    for activity in range(4):
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
        activity_index = activity
        x_train = x_train[act_train_labels[:, activity_index]==1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity_index]==1]
        act_train_labels = act_train_labels[act_train_labels[:, activity_index]==1]
        act_train = act_train[act_train == activity_index]
                
        y_train = np.zeros((gen_train_labels.shape[0], 2))
        for i in range(gen_train_labels.shape[0]):
            count = 0
            gen = gen_train_labels[i]
            if gen == 0:
                y_train[i, 0] = 1
            else:
                y_train[i, 1] = 1
        
        y_train = y_train[np.logical_or.reduce((act_train == 0., act_train == 1., act_train == 2., act_train == 3.))]

        tensor_x = torch.from_numpy(x_train.astype('float32')) # transform to torch tensor
        tensor_y = torch.from_numpy(y_train.astype('float32'))
        vae_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = torch.utils.data.DataLoader(vae_dataset, batch_size=512, shuffle=True)

        aux = AUX(z_dim, numLabels=2)
        if use_gpu:
            aux.cuda(idgpu)
        encodermodel = Encoder()
        if usecuda:
            encodermodel.cuda(idgpu)
        decodermodel = Decoder()
        if usecuda:
            decodermodel.cuda(idgpu)

        optimizerencoder = optim.Adam(encodermodel.parameters())
        optimizerdecoder = optim.Adam(decodermodel.parameters())
        optimizer_aux = optim.Adam(aux.parameters())

        for i in range(200):
            for batch_idx, (train_x, train_y) in enumerate(train_loader):
                train_x= Variable(train_x)
                train_y= Variable(train_y)

                true_samples = torch.randn((len(train_x),z_dim))
                true_samples = Variable(true_samples)

                if(usecuda):
                    train_x = train_x.cuda(idgpu)
                    true_samples = true_samples.cuda(idgpu)
                    train_y = train_y.cuda(idgpu)

                optimizerencoder.zero_grad()
                optimizerdecoder.zero_grad()
                optimizer_aux.zero_grad()

                # cat_x = torch.cat((train_x, train_y), dim = 1)
                train_z, mu, log_var = encodermodel(train_x)
                cat_z = torch.cat((train_z, train_y), dim = 1)
                train_xr = decodermodel(cat_z)

                for k in range(20):
                    #Train the aux net to predict y from z
                    auxY = aux(train_z.detach()) #detach: to ONLY update the AUX net #the prediction here for GT being predY
                    auxLoss = F.binary_cross_entropy(auxY.type_as(train_y), train_y) #correct order  #predY is a Nx2 use 2nd col.
                    auxLoss.backward()
                    optimizer_aux.step()

                #Train the encoder to NOT predict y from z
                auxK = aux(train_z) #not detached update the encoder!
                auxEncLoss = F.binary_cross_entropy(auxK.type_as(train_y), train_y)
                vaeLoss = -auxEncLoss

                recons_loss = F.mse_loss(train_xr, train_x)*512
                kld_loss = torch.mean(-2* torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = (recons_loss + kld_loss)/150 + 0.2*vaeLoss
                loss.backward()

                optimizerencoder.step()
                optimizerdecoder.step()

                if(batch_idx%100 == 0):
                    print("Epoch %d : MSE is %f, KLD loss is %f, AUX loss is %f" % (i,recons_loss.data, kld_loss.data, auxLoss))
        torch.save(encodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_motion_g_encoder_alpha_02_beta_2_'+str(activity)+str(z_dim))
        torch.save(decodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_motion_g_decoder_alpha_02_beta_2_'+str(activity)+str(z_dim))

def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

Latent_means = np.zeros((5, 2, z_dim))

#Used for F1-score
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


encodermodel_0 = Encoder().double()
encodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_encoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    encodermodel_0.cuda(idgpu)
decodermodel_0 = Decoder().double()
decodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_decoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    decodermodel_0.cuda(idgpu)

encodermodel_1 = Encoder().double()
encodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_encoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    encodermodel_1.cuda(idgpu)
decodermodel_1 = Decoder().double()
decodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_decoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    decodermodel_1.cuda(idgpu)

encodermodel_2 = Encoder().double()
encodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_encoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    encodermodel_2.cuda(idgpu)
decodermodel_2 = Decoder().double()
decodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_decoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    decodermodel_2.cuda(idgpu)

encodermodel_3 = Encoder().double()
encodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_encoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    encodermodel_3.cuda(idgpu)
decodermodel_3 = Decoder().double()
decodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_motion_g_decoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    decodermodel_3.cuda(idgpu)

#Load data and perform Condition Manipulation of the test data
for activity_index in range(4):
    print("This is the current activity")
    print(activity_index)
    # Testing
    train_data = DS.all_test()
    act_train_labels = DS.all_test_labels()
    gen_train_labels = DS.all_gender_test_labels()
    act_train = DS.get_act_test()
    train_data = train_data[act_train_labels[:, activity_index]==1]
    gen_train_labels = gen_train_labels[act_train_labels[:, activity_index]==1]
    act_train_labels = act_train_labels[act_train_labels[:, activity_index]==1]
    act_train = act_train[act_train == activity_index]

    eval_act_model = load_model("activity_model_mlp.hdf5")
    eval_gen_model = load_model("gender_model_mlp.hdf5")

    X = train_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)

    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    pred_act = np.zeros((train_data.shape[0],4))
    pred_gen = np.zeros((train_data.shape[0],2))

    Y_act = eval_act_model.predict(X)

    for index in range(train_data.shape[0]):
        index_act = np.argmax(Y_act[index], axis=0)
        pred_act[index, index_act] = 1
    
    Y_gen = eval_gen_model.predict(X)

    for index in range(train_data.shape[0]):
        if Y_gen[index] > 0.5:
            pred_gen[index, 1] = 1
        else:
            pred_gen[index, 0] = 1
    
    hat_train_data = np.empty((0,256), float)
    hat_gen_data = np.empty((0), float)

    for act_inside in range(4):
        print(act_inside)
        Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
        X_inside = train_data[pred_act[:, act_inside] == 1]
        Y_gen_inside = pred_gen[pred_act[:, act_inside] == 1]
        Y_test_gen = gen_train_labels[pred_act[:, act_inside] == 1]
                
        if Y_act_inside != []:
            if act_inside == 0:
                encodermodel = encodermodel_0
                decodermodel = decodermodel_0
            elif act_inside == 1:
                encodermodel = encodermodel_1
                decodermodel = decodermodel_1
            elif act_inside == 2:
                encodermodel = encodermodel_2
                decodermodel = decodermodel_2
            elif act_inside == 3:
                encodermodel = encodermodel_3
                decodermodel = decodermodel_3

            X_inside = np.reshape(X_inside, [X_inside.shape[0], 256])
            tensor_X = torch.from_numpy(X_inside) # transform to torch tensor
            z = np.empty((0,z_dim), float)

            y_dataset = np.zeros((Y_gen_inside.shape[0], 2))
            count = 0
            for i in range(Y_test_gen.shape[0]):
                if Y_test_gen[i] == 0:
                    count = count + 1
                else:
                    pass
            count = 0
            for i in range(Y_gen_inside.shape[0]):
                if Y_gen_inside[i, 0] == 1:
                    count = count + 1
                    y_dataset[i, 0] = 1
                else:
                    y_dataset[i, 1] = 1
            tensor_Y = torch.from_numpy(y_dataset)
            data_dataset = TensorDataset(tensor_X, tensor_Y)
            train_loader = torch.utils.data.DataLoader(data_dataset, batch_size=256, shuffle=False)

            for batch_idx, (x, y) in enumerate(train_loader):
                x= Variable(x)
                y = Variable(y)
                if(usecuda):
                    x = x.cuda(idgpu)
                    y = y.cuda(idgpu)
                z_e = encodermodel(x)[0]
                z = np.append(z, z_e.data.cpu(), axis=0)

            z_train = z.copy()

            tensor_z = torch.from_numpy(z_train) # transform to torch tensor
            y_dataset = np.zeros((Y_gen_inside.shape[0], 2))
            for i in range(Y_gen_inside.shape[0]):
                if Y_gen_inside[i, 0] == 1:
                    y_dataset[i, 1] = 1
                else:
                    y_dataset[i, 0] = 1
            tensor_y = torch.from_numpy(y_dataset)

            z_dataset = TensorDataset(tensor_z, tensor_y)
            z_loader = torch.utils.data.DataLoader(z_dataset, batch_size=256, shuffle=False)

            for batch_idx, (z, y) in enumerate(z_loader):
                z = Variable(z)
                y = Variable(y)
                if(use_gpu):
                    z = z.cuda(idgpu)
                    y = y.cuda(idgpu)
                z_cat = torch.cat((z, y), dim=1)
                x_hat = decodermodel(z_cat)
                hat_train_data = np.append(hat_train_data, x_hat.data.cpu(), axis=0)
            hat_gen_data = np.append(hat_gen_data, Y_test_gen, axis=0)

    reconstructed_input = hat_train_data
    X = reconstructed_input
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)
    
    X = hat_train_data
    X_all = np.append(X_all, X, axis=0)

    Y = act_train_labels
    Y_all_act = np.append(Y_all_act, Y, axis=0)

    Y = hat_gen_data
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    Y = hat_gen_data
    Y_all_gen = np.append(Y_all_gen, Y, axis=0)

print_act_results_f1_score(eval_act_model, X_all, Y_all_act)
print_gen_results_f1_score(eval_gen_model, X_all, Y_all_gen)