#20 re-inference accuracy for deterministic, probabilistic and randomized manipulation Figure 4

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
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
from sklearn.model_selection import train_test_split

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
x_dim = 2 #size of condition variable
zed = [5] #size of latent representation

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

def eval_test(X, Y, File="eval_gen_reid.hdf5", gen_class_numbers=1, fm=(2, 5), ep=50):
    height = X.shape[1]
    width = X.shape[2]
    # height = 1
    # width = 2
    ## Callbacks
    eval_metric = "val_acc"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=15)
    filepath = File
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=2, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]
    # eval_gen = Estimator.build(height, width, gen_class_numbers, name="EVAL_GEN", fm=fm, act_func="sigmoid",
    #                            hid_act_func="relu")
    eval_gen = Estimator.build(height, width, gen_class_numbers, name="EVAL_GEN", fm=fm, act_func="sigmoid",
                               hid_act_func="relu")
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    # X, Y = shuffle(X, Y)
    eval_gen.fit(X, Y,
                 # validation_split=.1,
                 epochs=ep,
                 batch_size=512,
                 verbose=1,
                 class_weight=get_class_weights(Y)
                 # callbacks=callbacks_list
                 )
    eval_gen.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    eval_gen.save(File)

for z_dim in zed:
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(768, 512)
            self.fc2 = nn.Linear(512,512)
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
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
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
            self.fc4 = nn.Linear(512,512)
            self.fc5 = nn.Linear(512, 768)
            self.relu = nn.ReLU()

        def forward(self,x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
            h5 = self.fc5(h4)
            return h5

    class AUX(nn.Module):
        def __init__(self, nz, numLabels=1):
            super(AUX, self).__init__()
            self.nz = nz
            self.numLabels = numLabels
            self.aux1 = nn.Linear(nz, 128)
            # self.aux2 = nn.Linear(128, 64)
            self.aux3 = nn.Linear(128, numLabels)
        def infer_y_from_z(self, z):
            z = F.relu(self.aux1(z))
            # z = F.relu(self.aux2(z))
            if self.numLabels==1:
                z = F.sigmoid(self.aux3(z))
            else:
                z = F.softmax(self.aux3(z))
            return z
        def forward(self, z):
            return self.infer_y_from_z(z)
        def loss(self, pred, target):
            return F.nll_loss(pred, target)

    data_subjects = pd.read_csv("/home/omid/pycharm/Mobi/data_subjects.csv")

    data = np.load("Data/total_data.npy", allow_pickle=True)
    activity = np.load("Data/activity_labels.npy", allow_pickle=True)
    gender = np.load("Data/gender_labels.npy", allow_pickle=True)
    age = np.load("Data/age_labels.npy", allow_pickle=True)
    id = np.load("Data/id_labels.npy", allow_pickle=True)

    array = np.arange(data.shape[0])
    np.random.shuffle(array)

    data = data[array]
    activity = activity[array]
    gender = gender[array]
    age = age[array]
    id = id[array]

    data_train = np.array([]).reshape(0, data.shape[1], data.shape[2])
    data_test = np.array([]).reshape(0, data.shape[1], data.shape[2])
    activity_train = np.array([])
    activity_test = np.array([])
    age_train = np.array([])
    age_test = np.array([])
    gender_train = np.array([])
    gender_test = np.array([])

    for i in data_subjects["id"]:
        data_sub_id = data[id[:] == i]
        age_sub_id = age[id[:] == i]
        activity_sub_id = activity[id[:] == i]
        gender_sub_id = gender[id[:] == i]
        x_train, x_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(data_sub_id, age_sub_id, activity_sub_id, gender_sub_id, test_size = 0.2, random_state = 42)
        data_train = np.concatenate((data_train, x_train), axis=0)
        data_test = np.concatenate((data_test, x_test), axis=0)
        age_train = np.concatenate((age_train, y_train), axis=0)
        age_test = np.concatenate((age_test, y_test), axis=0)
        activity_train = np.concatenate((activity_train, z_train), axis=0)
        activity_test = np.concatenate((activity_test, z_test), axis=0)
        gender_train = np.concatenate((gender_train, w_train), axis=0)
        gender_test = np.concatenate((gender_test, w_test), axis=0)

    nb_classes = len(np.unique(activity_train[:]))
    activity_train_label = keras.utils.to_categorical(activity_train[:], nb_classes)
    nb_classes = len(np.unique(activity_test[:]))
    activity_test_label = keras.utils.to_categorical(activity_test[:], nb_classes)
    nb_classes = len(np.unique(age_train[:]))
    age_train_label = keras.utils.to_categorical(age_train[:], nb_classes)
    nb_classes = len(np.unique(age_test[:]))
    age_test_label = keras.utils.to_categorical(age_test[:], nb_classes)
    gender_train_label = gender_train
    gender_test_label = gender_test
    x_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
    x_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

    for activity in range(4):
        print("########################################################")
        print(activity)
        x_vae = x_train[activity_train_label[:, activity] == 1]
        act_vae = activity_train_label[activity_train_label[:, activity] == 1]
        gen_vae = gender_train_label[activity_train_label[:, activity] == 1]
        x_vae_size = x_vae.shape[0]
        x_vae = np.reshape(x_vae, [x_vae_size, 768])
        gen_train = np.zeros((gen_vae.shape[0], 2))
        for i in range(gen_train.shape[0]):
            count = 0
            gen = gen_vae[i]
            if gen == 0:
                gen_train[i, 0] = 1
            else:
                gen_train[i, 1] = 1
        y = np.zeros((gen_vae.shape[0], 1))
        for i in range(gen_train.shape[0]):
            count = 0
            gen = gen_vae[i]
            y[i, 0] = gen
        
        tensor_x = torch.from_numpy(x_vae.astype('float32')) # transform to torch tensor
        tensor_y = torch.from_numpy(gen_train.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))

        vae_dataset = TensorDataset(tensor_x, tensor_y, y)
        train_loader = torch.utils.data.DataLoader(vae_dataset, batch_size=128, shuffle=True)
        aux = AUX(z_dim, numLabels=1)
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
'''
        for i in range(200):
            for batch_idx, (train_x, train_y, y) in enumerate(train_loader):
                train_x= Variable(train_x)
                train_y= Variable(train_y)
                y = Variable(y)
                true_samples = torch.randn((len(train_x),z_dim))
                true_samples = Variable(true_samples)

                if(usecuda):
                    train_x = train_x.cuda(idgpu)
                    true_samples = true_samples.cuda(idgpu)
                    train_y = train_y.cuda(idgpu)
                    y = y.cuda(idgpu)

                optimizerencoder.zero_grad()
                optimizerdecoder.zero_grad()
                optimizer_aux.zero_grad()

                train_z, mu, log_var = encodermodel(train_x)
                cat_z = torch.cat((train_z, train_y), dim = 1)
                train_xr = decodermodel(cat_z)

                for k in range(20):
                    #Train the aux net to predict y from z
                    auxY = aux(train_z.detach()) #detach: to ONLY update the AUX net #the prediction here for GT being predY
                    auxLoss = F.binary_cross_entropy(auxY.type_as(y), y) #correct order  #predY is a Nx2 use 2nd col.
                    auxLoss.backward()
                    optimizer_aux.step()

                #Train the encoder to NOT predict y from z
                auxK = aux(train_z) #not detached update the encoder!
                auxEncLoss = F.binary_cross_entropy(auxK.type_as(y), y)
                vaeLoss = -auxEncLoss

                recons_loss = F.mse_loss(train_xr, train_x)*512
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = torch.mean((recons_loss + kld_loss)/150) + vaeLoss
                loss.backward()

                # optimizer_aux.zero_grad()
                optimizerencoder.step()
                optimizerdecoder.step()

                if(batch_idx%100 == 0):
                    print("Epoch %d : MSE is %f, KLD loss is %f, AUX loss is %f" % (i,recons_loss.data, kld_loss.data, auxLoss))

        torch.save(encodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_'+str(activity)+str(z_dim))
        torch.save(decodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_'+str(activity)+str(z_dim))
'''
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

z_dim = 5
def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

Latent_means = np.zeros((5, 2, z_dim))

act_label = 0
gen_label = 0
data_subjects = pd.read_csv("/home/omid/pycharm/Mobi/data_subjects.csv")
data = np.load("/home/omid/pycharm/Mobi/Data/total_data.npy", allow_pickle=True)
activity = np.load("/home/omid/pycharm/Mobi/Data/activity_labels.npy", allow_pickle=True)
gender = np.load("/home/omid/pycharm/Mobi/Data/gender_labels.npy", allow_pickle=True)
age = np.load("/home/omid/pycharm/Mobi/Data/age_labels.npy", allow_pickle=True)
id = np.load("/home/omid/pycharm/Mobi/Data/id_labels.npy", allow_pickle=True)

array = np.arange(data.shape[0])
np.random.shuffle(array)
data = data[array]
activity = activity[array]
gender = gender[array]
age = age[array]
id = id[array]

data_train = np.array([]).reshape(0, data.shape[1], data.shape[2])
data_test = np.array([]).reshape(0, data.shape[1], data.shape[2])
activity_train = np.array([])
activity_test = np.array([])
age_train = np.array([])
age_test = np.array([])
gender_train = np.array([])
gender_test = np.array([])

for i in data_subjects["id"]:
    data_sub_id = data[id[:] == i]
    age_sub_id = age[id[:] == i]
    activity_sub_id = activity[id[:] == i]
    gender_sub_id = gender[id[:] == i]
    x_train, x_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(data_sub_id, age_sub_id, activity_sub_id, gender_sub_id, test_size = 0.2, random_state = 42)
    data_train = np.concatenate((data_train, x_train), axis=0)
    data_test = np.concatenate((data_test, x_test), axis=0)
    age_train = np.concatenate((age_train, y_train), axis=0)
    age_test = np.concatenate((age_test, y_test), axis=0)
    activity_train = np.concatenate((activity_train, z_train), axis=0)
    activity_test = np.concatenate((activity_test, z_test), axis=0)
    gender_train = np.concatenate((gender_train, w_train), axis=0)
    gender_test = np.concatenate((gender_test, w_test), axis=0)

nb_classes = len(np.unique(activity_train[:]))
activity_train_label = keras.utils.to_categorical(activity_train[:], nb_classes)
nb_classes = len(np.unique(activity_test[:]))
activity_test_label = keras.utils.to_categorical(activity_test[:], nb_classes)
nb_classes = len(np.unique(age_train[:]))
age_train_label = keras.utils.to_categorical(age_train[:], nb_classes)
nb_classes = len(np.unique(age_test[:]))
age_test_label = keras.utils.to_categorical(age_test[:], nb_classes)
gender_train_label = gender_train
gender_test_label = gender_test
x_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
x_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

encodermodel_0 = Encoder().double()
encodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    encodermodel_0.cuda(idgpu)
decodermodel_0 = Decoder().double()
decodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    decodermodel_0.cuda(idgpu)

encodermodel_1 = Encoder().double()
encodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    encodermodel_1.cuda(idgpu)
decodermodel_1 = Decoder().double()
decodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    decodermodel_1.cuda(idgpu)

encodermodel_2 = Encoder().double()
encodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    encodermodel_2.cuda(idgpu)
decodermodel_2 = Decoder().double()
decodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    decodermodel_2.cuda(idgpu)

encodermodel_3 = Encoder().double()
encodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    encodermodel_3.cuda(idgpu)
decodermodel_3 = Decoder().double()
decodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    decodermodel_3.cuda(idgpu)

accuracy_reinference = np.zeros((20, 3)) #20 re-inference accuracy for deterministic, probabilistic and randomized manipulation

for looooop in range(20): #20 re-inference accuracy for deterministic, probabilistic and randomized manipulation
    print("Probabilistic Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_gen = np.empty([0])

    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_gen_model = load_model("gender_model_DC.hdf5")
    X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    Y = act_train_labels
    print("Activity Identification")
    print_results(eval_act_model, X, Y)
    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("Gender Identification " + str(act_acc))

    np.save("/home/omid/pycharm/Mobi/data/mobi_g_org", X)

    import secrets
    for activity in range(4):
        print("This is the current activity")
        print(activity)
        # TESTing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        ### Manipulation at the Gender Level
        X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        Y = gen_train_labels

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
        
        hat_train_data = np.empty((0,768), float)
        hat_gen_data = np.empty((0), float)

        for act_inside in range(4):
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

                X_inside = np.reshape(X_inside, [X_inside.shape[0], 768])
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
                z = np.empty((0,z_dim), float)

                for batch_idx, (train_x, train_y) in enumerate(train_loader):
                    train_x= Variable(train_x)
                    train_y= Variable(train_y)
                    if(usecuda):
                        train_x = train_x.cuda(idgpu)
                        train_y = train_y.cuda(idgpu)
                    z_batch = encodermodel(train_x)[0]
                    z = np.append(z, z_batch.data.cpu(), axis=0)

                z_train = z.copy()

                tensor_z = torch.from_numpy(z_train) # transform to torch tensor
                y_dataset = np.zeros((Y_gen_inside.shape[0], 2))

                for i in range(Y_gen_inside.shape[0]):
                    rand_1 = secrets.randbelow(100)/100
                    if 0.5 < rand_1:
                        if Y_gen_inside[i, 0] == 1:
                            y_dataset[i, 1] = 1
                        else:
                            y_dataset[i, 0] = 1
                    else:
                        if Y_gen_inside[i, 0] == 1:
                            y_dataset[i, 0] = 1
                        else:
                            y_dataset[i, 1] = 1
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
        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)

        Y = hat_gen_data
        Y_all_gen = np.append(Y_all_gen, Y, axis=0)

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_gen_model.evaluate(X_all, Y_all_gen)
    act_gen = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_gen))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_gen, test_size=0.2, random_state=42)

    eval_test(X_te, y_te, File="eval_gen_reid.hdf5")
    eval_gen_reid = load_model("eval_gen_reid.hdf5")
    result1 = eval_gen_reid.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Gender Identification " + str(act_acc))
    accuracy_reinference[looooop, 1] = act_acc

    print()
    print()
    print("Deterministic Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_gen = np.empty([0])

    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_gen_model = load_model("gender_model_DC.hdf5")
    X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    Y = act_train_labels
    print("Activity Identification")
    print_results(eval_act_model, X, Y)
    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("Gender Identification " + str(act_acc))

    import secrets
    for activity in range(4):
        print("This is the current activity")
        print(activity)
        # TESTing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        # activity_index = 0
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        Y = gen_train_labels

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
        
        hat_train_data = np.empty((0,768), float)
        hat_gen_data = np.empty((0), float)
        for act_inside in range(4):
            # print(act_inside)
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

                X_inside = np.reshape(X_inside, [X_inside.shape[0], 768])
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
                z = np.empty((0,z_dim), float)

                for batch_idx, (train_x, train_y) in enumerate(train_loader):
                    train_x= Variable(train_x)
                    train_y= Variable(train_y)
                    if(usecuda):
                        train_x = train_x.cuda(idgpu)
                        train_y = train_y.cuda(idgpu)
                    z_batch = encodermodel(train_x)[0]
                    z = np.append(z, z_batch.data.cpu(), axis=0)

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
        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)
        Y = hat_gen_data
        Y_all_gen = np.append(Y_all_gen, Y, axis=0)

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_gen_model.evaluate(X_all, Y_all_gen)
    act_gen = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_gen))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_gen, test_size=0.2, random_state=42)

    eval_test(X_te, y_te, File="eval_gen_reid.hdf5")
    eval_gen_reid = load_model("eval_gen_reid.hdf5")
    result1 = eval_gen_reid.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Gender redentification " + str(act_acc))
    accuracy_reinference[looooop, 0] = act_acc

    print()
    print()
    print("RandomVector Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_gen = np.empty([0])

    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_gen_model = load_model("gender_model_DC.hdf5")
    X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    Y = act_train_labels
    print("Activity Identification")
    print_results(eval_act_model, X, Y)
    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("Gender Identification " + str(act_acc))

    import secrets
    for activity in range(4):
        print("This is the current activity")
        print(activity)
        # TESTing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        ### Manipulation at the Gender Level
        X = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        Y = gen_train_labels

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
        
        hat_train_data = np.empty((0,768), float)
        hat_gen_data = np.empty((0), float)

        for act_inside in range(4):
            # print(act_inside)
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

                X_inside = np.reshape(X_inside, [X_inside.shape[0], 768])
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
                z = np.empty((0,z_dim), float)

                for batch_idx, (train_x, train_y) in enumerate(train_loader):
                    train_x= Variable(train_x)
                    train_y= Variable(train_y)
                    if(usecuda):
                        train_x = train_x.cuda(idgpu)
                        train_y = train_y.cuda(idgpu)
                    z_batch = encodermodel(train_x)[0]
                    z = np.append(z, z_batch.data.cpu(), axis=0)

                z_train = z.copy()

                tensor_z = torch.from_numpy(z_train) # transform to torch tensor
                y_dataset = np.zeros((Y_gen_inside.shape[0], 2))
                
                for i in range(Y_gen_inside.shape[0]):
                    rand_1 = secrets.randbelow(100)/100
                    y_dataset[i, 0] = rand_1
                    y_dataset[i, 1] = 1 - rand_1
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

        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)

        Y = hat_gen_data
        Y_all_gen = np.append(Y_all_gen, Y, axis=0)

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_gen_model.evaluate(X_all, Y_all_gen)
    act_gen = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_gen))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_gen, test_size=0.2, random_state=42)

    eval_test(X_te, y_te, File="eval_gen_reid.hdf5")
    eval_gen_reid = load_model("eval_gen_reid.hdf5")
    result1 = eval_gen_reid.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Gender reidentification " + str(act_acc))
    accuracy_reinference[looooop, 2] = act_acc

df = pd.DataFrame(data=accuracy_reinference, columns=["det", "prob", "randvect"])
df.to_csv('reid_mobi_g.csv', index=False)