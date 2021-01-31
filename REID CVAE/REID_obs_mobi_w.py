import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"] = "1";
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
import secrets
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
x_dim = 3
AI = 1 #Activity Index
zed = [5]
ma_rate = 0.001

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

def eval_weight(X, Y, weight_class_numbers=3, fm=(2, 5), ep=50):
    height = X.shape[1]
    width = X.shape[2]
    ## Callbacks
    eval_metric = "val_acc"
    filepath = "age_model"
    early_stop = keras.callbacks.EarlyStopping(monitor=eval_metric, mode='max', patience=20)
    checkpoint = ModelCheckpoint(filepath, monitor=eval_metric, verbose=0, save_best_only=True, mode='max')
    callbacks_list = [early_stop, checkpoint]
    # eval_act = Estimator.build(height, width, act_class_numbers, name="EVAL_ACT", fm=fm, act_func="softmax",
    #                            hid_act_func="relu")
    eval_weight = Estimator.build(height, width, weight_class_numbers, name="EVAL_ACT", fm=fm, act_func="softmax",
                               hid_act_func="relu")
    eval_weight.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    # X, Y = shuffle(X, Y)
    eval_weight.fit(X, Y,
                 validation_split=.1,
                 epochs=ep,
                 batch_size=256,
                 verbose=1,
                 shuffle=True,
                 class_weight=get_class_weights(np.argmax(Y, axis=1))
                 # callbacks=callbacks_list
                 )
    eval_weight.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print_results(eval_weight, X, Y)
    eval_weight.save("weight_model_test.hdf5")

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

    class LatentSmoothing(nn.Module):
        def __init__(self):
            super(LatentSmoothing, self).__init__()
            self.fc1 = nn.Linear(z_dim, 2)
            self.softmax = nn.Softmax()

        def forward(self,x):
            h1 = self.fc1(x)
            return h1

    class Mine(nn.Module):
        def __init__(self, z_size=z_dim, gender_size=1, output_size=1, hidden_size=128):
            super().__init__()
            self.fc1_noise = nn.Linear(z_size, hidden_size, bias=False)
            self.fc1_sample = nn.Linear(gender_size, hidden_size, bias=False)
            self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.ma_et = None
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        def forward(self, z, gender):
            x_z = self.fc1_noise(z)
            x_gender = self.fc1_sample(gender)
            x = F.leaky_relu(x_z + x_gender + self.fc1_bias, negative_slope=2e-1)
            x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
            x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
            return x
    
    class AUX(nn.Module):
        def __init__(self, nz, numLabels=3):
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
    
    def compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
        tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

    def compute_mmd(x, y):
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

    data_subjects = pd.read_csv("/home/omid/pycharm/Mobi/data_subjects.csv")
    data = np.load("Data/total_data.npy", allow_pickle=True)
    activity = np.load("Data/activity_labels.npy", allow_pickle=True)
    gender = np.load("Data/gender_labels.npy", allow_pickle=True)
    age = np.load("Data/age_labels.npy", allow_pickle=True)
    id = np.load("Data/id_labels.npy", allow_pickle=True)
    weight = np.load("Data/weights_data.npy", allow_pickle=True)

    array = np.arange(data.shape[0])
    np.random.shuffle(array)

    data = data[array]
    print(data.shape)
    activity = activity[array]
    gender = gender[array]
    print(weight)
    weight = weight[array]
    print(weight)
    print("weights printed")
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
    weight_train = np.array([])
    weight_test = np.array([])

    for i in data_subjects["id"]:
        data_sub_id = data[id[:] == i]
        age_sub_id = age[id[:] == i]
        activity_sub_id = activity[id[:] == i]
        gender_sub_id = gender[id[:] == i]
        weight_sub_id = weight[id[:] == i]
        x_train, x_test, y_train, y_test, z_train, z_test, w_train, w_test, t_train, t_test = train_test_split(data_sub_id, age_sub_id, activity_sub_id, gender_sub_id, weight_sub_id, test_size = 0.2, random_state = 42)
        data_train = np.concatenate((data_train, x_train), axis=0)
        data_test = np.concatenate((data_test, x_test), axis=0)
        age_train = np.concatenate((age_train, y_train), axis=0)
        age_test = np.concatenate((age_test, y_test), axis=0)
        activity_train = np.concatenate((activity_train, z_train), axis=0)
        activity_test = np.concatenate((activity_test, z_test), axis=0)
        gender_train = np.concatenate((gender_train, w_train), axis=0)
        gender_test = np.concatenate((gender_test, w_test), axis=0)
        weight_train = np.concatenate((weight_train, t_train), axis=0)
        weight_test = np.concatenate((weight_test, t_test), axis=0)
    
    for i in range(weight_train.shape[0]):
        if weight_train[i] <= 70:
            weight_train[i] = 0
        elif weight_train[i] <= 90:
            weight_train[i] = 1
        else:
            weight_train[i] = 2
    
    for i in range(weight_test.shape[0]):
        if weight_test[i] <= 70:
            weight_test[i] = 0
        elif weight_test[i] <= 90:
            weight_test[i] = 1
        else:
            weight_test[i] = 2
    print(weight_train)
    print(weight_test)

    def get_class_weights(y):
        counter = Counter(y)
        majority = max(counter.values())
        return {cls: float(majority / count) for cls, count in counter.items()}
    
    nb_classes = len(np.unique(activity_train[:]))
    activity_train_label = keras.utils.to_categorical(activity_train[:], nb_classes)
    nb_classes = len(np.unique(activity_test[:]))
    activity_test_label = keras.utils.to_categorical(activity_test[:], nb_classes)
    nb_classes = len(np.unique(age_train[:]))
    age_train_label = keras.utils.to_categorical(age_train[:], nb_classes)
    nb_classes = len(np.unique(age_test[:]))
    age_test_label = keras.utils.to_categorical(age_test[:], nb_classes)
    nb_classes = len(np.unique(weight_train[:]))
    weight_train_label = keras.utils.to_categorical(weight_train[:], nb_classes)
    nb_classes = len(np.unique(weight_test[:]))
    weight_test_label = keras.utils.to_categorical(weight_test[:], nb_classes)
    gender_train_label = gender_train
    gender_test_label = gender_test
    x_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
    x_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

    for activity in [1]:

        x_vae = x_train[activity_train_label[:, activity] == 1]
        act_vae = activity_train_label[activity_train_label[:, activity] == 1]
        weight_vae = weight_train_label[activity_train_label[:, activity] == 1]

        x_vae_size = x_vae.shape[0]
        x_vae = np.reshape(x_vae, [x_vae_size, 768])
        
        tensor_x = torch.from_numpy(x_vae.astype('float32')) # transform to torch tensor
        tensor_y = torch.from_numpy(weight_vae.astype('float32'))
        vae_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = torch.utils.data.DataLoader(vae_dataset, batch_size=512, shuffle=True)

        aux = AUX(z_dim, numLabels=3)
        if use_gpu:
            aux.cuda()
        
        encodermodel = Encoder()
        if usecuda:
            encodermodel.cuda()
        
        decodermodel = Decoder()
        if usecuda:
            decodermodel.cuda()

        optimizerencoder = optim.Adam(encodermodel.parameters())
        optimizerdecoder = optim.Adam(decodermodel.parameters())
        optimizer_aux = optim.Adam(aux.parameters())
'''
        for i in range(200):
            for batch_idx, (train_x, train_y) in enumerate(train_loader):
                train_x= Variable(train_x)
                train_y= Variable(train_y)
                true_samples = torch.randn((len(train_x),z_dim))
                true_samples = Variable(true_samples)

                if(usecuda):
                    train_x = train_x.cuda()
                    true_samples = true_samples.cuda()
                    train_y = train_y.cuda()

                optimizerencoder.zero_grad()
                optimizerdecoder.zero_grad()
                optimizer_aux.zero_grad()

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
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = torch.mean((recons_loss + kld_loss)/150) + 2*vaeLoss
                loss.backward()

                optimizerencoder.step()
                optimizerdecoder.step()

                if(batch_idx%100 == 0):
                    # result1 = aux(train_z)
                    # _, result1 = torch.max(result1.data, 1)
                    # correct = (result1 == train_y).float().sum()
                    # accuracy = 100 * correct / (train_z.shape[0])
                    # print("***[RESULT]*** Aux results" + str(accuracy.data))
                    print("Epoch %d : MSE is %f, KLD loss is %f, AUX loss is %f" % (i,recons_loss.data, kld_loss.data, auxLoss.data))
        
        torch.save(encodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_w_encoder_'+str(activity)+str(z_dim))
        torch.save(decodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_w_decoder_'+str(activity)+str(z_dim))
'''
def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

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
encodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_encoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    encodermodel_0.cuda()
decodermodel_0 = Decoder().double()
decodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_decoder_alpha_02_beta_2_0'+str(z_dim)))
if usecuda:
    decodermodel_0.cuda()

encodermodel_1 = Encoder().double()
encodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_encoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    encodermodel_1.cuda()
decodermodel_1 = Decoder().double()
decodermodel_1.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_decoder_alpha_02_beta_2_1'+str(z_dim)))
if usecuda:
    decodermodel_1.cuda()

encodermodel_2 = Encoder().double()
encodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_encoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    encodermodel_2.cuda()
decodermodel_2 = Decoder().double()
decodermodel_2.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_decoder_alpha_02_beta_2_2'+str(z_dim)))
if usecuda:
    decodermodel_2.cuda()

encodermodel_3 = Encoder().double()
encodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_encoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    encodermodel_3.cuda()
decodermodel_3 = Decoder().double()
decodermodel_3.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/obs_mobi_w_decoder_alpha_02_beta_2_3'+str(z_dim)))
if usecuda:
    decodermodel_3.cuda()

accuracy_reinference = np.zeros((20, 3))

for looooop in range(2):
    print()
    print()
    print("Deterministic Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_weight = np.empty([0, 3])

    # Testing
    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    age_train_labels = age_test_label
    weight_train_labels = weight_test_label

    # activity_index = 0
    # train_data = train_data[act_train_labels[:, activity] == 1]
    # gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
    # age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
    # weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
    # act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_weight_model = load_model("weight_model_DC.hdf5")

    t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    X = t_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)
    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = weight_train_labels
    result1 = eval_weight_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    for activity in range(4):
        print("This is the current activity")
        print(activity)

        # Testing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        age_train_labels = age_test_label
        weight_train_labels = weight_test_label

        # activity_index = 0
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
        weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        X = t_data
        Y = act_train_labels
        # print("Activity Identification for Gender 0")
        # print_results(eval_act_model, X, Y)
        # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
        Y = weight_train_labels
        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))
        
        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        pred_act = np.zeros((train_data.shape[0], 4))
        pred_weight = np.zeros((train_data.shape[0], 3))

        X = train_data
        Y_act = eval_act_model.predict(X)
        for index in range(train_data.shape[0]):
            index_act = np.argmax(Y_act[index], axis=0)
            pred_act[index, index_act] = 1
        
        Y_weight = eval_weight_model.predict(X)
        for index in range(train_data.shape[0]):
            gen = np.argmax(Y_weight[index], axis=0)
            if gen == 0:
                pred_weight[index, 0] = 1
            elif gen == 1:
                pred_weight[index, 1] = 1
            else:
                pred_weight[index, 2] = 1
        
        hat_train_data = np.empty((0,768), float)
        hat_wght_data = np.empty((0,3), float)

        for act_inside in range(4):
            print(act_inside)
            Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
            X_inside = train_data[pred_act[:, act_inside] == 1]
            Y_weight_inside = pred_weight[pred_act[:, act_inside] == 1]
            Y_test_wght = weight_train_labels[pred_act[:, act_inside] == 1]

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
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                
                for i in range(Y_weight_inside.shape[0]):
                    if Y_weight_inside[i, 0] == 1:
                        y_dataset[i, 0] = 1
                    elif Y_weight_inside[i, 1] == 1:
                        y_dataset[i, 1] = 1
                    elif Y_weight_inside[i, 2] == 1:
                        y_dataset[i, 2] = 1
                
                tensor_Y = torch.from_numpy(y_dataset)
                data_dataset = TensorDataset(tensor_X, tensor_Y)
                train_loader = torch.utils.data.DataLoader(data_dataset, batch_size=256, shuffle=False)

                for batch_idx, (x, y) in enumerate(train_loader):
                    x= Variable(x)
                    y = Variable(y)
                    if(usecuda):
                        x = x.cuda()
                        y = y.cuda()
                    # x_cat = torch.cat((x, y), dim=1)
                    z_e = encodermodel(x)[0]
                    z = np.append(z, z_e.data.cpu(), axis=0)

                z_train = z.copy()
                tensor_z = torch.from_numpy(z_train) # transform to torch tensor
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                for i in range(Y_weight_inside.shape[0]):
                    if Y_weight_inside[i, 0] == 1:
                        y_dataset[i, 2] = 1
                    elif Y_weight_inside[i, 1] == 1:
                        y_dataset[i, 0] = 1
                    elif Y_weight_inside[i, 2] == 1:
                        y_dataset[i, 1] = 1
                tensor_y = torch.from_numpy(y_dataset)

                z_dataset = TensorDataset(tensor_z, tensor_y)
                z_loader = torch.utils.data.DataLoader(z_dataset, batch_size=256, shuffle=False)

                for batch_idx, (z, y) in enumerate(z_loader):
                    z = Variable(z)
                    y = Variable(y)
                    if(use_gpu):
                        z = z.cuda()
                        y = y.cuda()
                    z_cat = torch.cat((z, y), dim=1)
                    x_hat = decodermodel(z_cat)
                    hat_train_data = np.append(hat_train_data, x_hat.data.cpu(), axis=0)
                hat_wght_data = np.append(hat_wght_data, Y_test_wght, axis=0)
        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        # print("Activity Identification:")
        # print_results(eval_act_model, X, Y)
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)

        Y = hat_wght_data
        Y_all_weight = np.append(Y_all_weight, Y, axis=0)

        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("Weight Identification: " + str(act_acc))

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_weight_model.evaluate(X_all, Y_all_weight)
    act_weight = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_weight))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_weight, test_size=0.2, random_state=42)

    eval_weight(X_te, y_te)
    eval_weight_test = load_model("weight_model_test.hdf5")
    result1 = eval_weight_test.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Weight re-Identification " + str(act_acc))
    accuracy_reinference[looooop, 0] = act_acc


    print()
    print()
    print("Probabilistic Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_weight = np.empty([0, 3])

    # Testing
    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    age_train_labels = age_test_label
    weight_train_labels = weight_test_label

    # activity_index = 0
    # train_data = train_data[act_train_labels[:, activity] == 1]
    # gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
    # age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
    # weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
    # act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_weight_model = load_model("weight_model_DC.hdf5")

    t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    X = t_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)
    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = weight_train_labels
    result1 = eval_weight_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    for activity in range(4):
        print("This is the current activity")
        print(activity)

        # Testing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        age_train_labels = age_test_label
        weight_train_labels = weight_test_label

        # activity_index = 0
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
        weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        X = t_data
        Y = act_train_labels
        # print("Activity Identification for Gender 0")
        # print_results(eval_act_model, X, Y)
        # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
        Y = weight_train_labels
        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))
        
        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        pred_act = np.zeros((train_data.shape[0], 4))
        pred_weight = np.zeros((train_data.shape[0], 3))

        X = train_data
        Y_act = eval_act_model.predict(X)
        for index in range(train_data.shape[0]):
            index_act = np.argmax(Y_act[index], axis=0)
            pred_act[index, index_act] = 1
        
        Y_weight = eval_weight_model.predict(X)
        for index in range(train_data.shape[0]):
            gen = np.argmax(Y_weight[index], axis=0)
            if gen == 0:
                pred_weight[index, 0] = 1
            elif gen == 1:
                pred_weight[index, 1] = 1
            else:
                pred_weight[index, 2] = 1
        
        hat_train_data = np.empty((0,768), float)
        hat_wght_data = np.empty((0,3), float)

        for act_inside in range(4):
            print(act_inside)
            Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
            X_inside = train_data[pred_act[:, act_inside] == 1]
            Y_weight_inside = pred_weight[pred_act[:, act_inside] == 1]
            Y_test_wght = weight_train_labels[pred_act[:, act_inside] == 1]
            
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
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                
                for i in range(Y_weight_inside.shape[0]):
                    if Y_weight_inside[i, 0] == 1:
                        y_dataset[i, 0] = 1
                    elif Y_weight_inside[i, 1] == 1:
                        y_dataset[i, 1] = 1
                    elif Y_weight_inside[i, 2] == 1:
                        y_dataset[i, 2] = 1
                
                tensor_Y = torch.from_numpy(y_dataset)
                data_dataset = TensorDataset(tensor_X, tensor_Y)
                train_loader = torch.utils.data.DataLoader(data_dataset, batch_size=256, shuffle=False)

                for batch_idx, (x, y) in enumerate(train_loader):
                    x= Variable(x)
                    y = Variable(y)
                    if(usecuda):
                        x = x.cuda()
                        y = y.cuda()
                    # x_cat = torch.cat((x, y), dim=1)
                    z_e = encodermodel(x)[0]
                    z = np.append(z, z_e.data.cpu(), axis=0)

                z_train = z.copy()
                tensor_z = torch.from_numpy(z_train) # transform to torch tensor
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                
                # rand_2 = secrets.randbelow(100)/100
                # rand_1 = secrets.randbelow(100)/100

                for i in range(Y_weight_inside.shape[0]):
                    rand_1 = secrets.randbelow(100)/100
                    if rand_1 > 0.5:
                        if Y_weight_inside[i, 0] == 1:
                            y_dataset[i, 2] = 1
                        elif Y_weight_inside[i, 1] == 1:
                            y_dataset[i, 0] = 1
                        elif Y_weight_inside[i, 2] == 1:
                            y_dataset[i, 1] = 1
                    else:
                        if Y_weight_inside[i, 0] == 1:
                            y_dataset[i, 0] = 1
                        elif Y_weight_inside[i, 1] == 1:
                            y_dataset[i, 1] = 1
                        elif Y_weight_inside[i, 2] == 1:
                            y_dataset[i, 2] = 1
                tensor_y = torch.from_numpy(y_dataset)

                z_dataset = TensorDataset(tensor_z, tensor_y)
                z_loader = torch.utils.data.DataLoader(z_dataset, batch_size=256, shuffle=False)

                for batch_idx, (z, y) in enumerate(z_loader):
                    z = Variable(z)
                    y = Variable(y)
                    if(use_gpu):
                        z = z.cuda()
                        y = y.cuda()
                    z_cat = torch.cat((z, y), dim=1)
                    x_hat = decodermodel(z_cat)
                    hat_train_data = np.append(hat_train_data, x_hat.data.cpu(), axis=0)
                hat_wght_data = np.append(hat_wght_data, Y_test_wght, axis=0)
        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        # print("Activity Identification:")
        # print_results(eval_act_model, X, Y)
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)

        Y = hat_wght_data
        Y_all_weight = np.append(Y_all_weight, Y, axis=0)

        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("Weight Identification: " + str(act_acc))

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_weight_model.evaluate(X_all, Y_all_weight)
    act_weight = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_weight))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_weight, test_size=0.2, random_state=42)

    eval_weight(X_te, y_te)
    eval_weight_test = load_model("weight_model_test.hdf5")
    result1 = eval_weight_test.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Weight re-Identification " + str(act_acc))
    accuracy_reinference[looooop, 1] = act_acc

    print()
    print()
    print("RandomVector Processing")

    X_all = np.empty([0, x_test.shape[1], x_test.shape[2],x_test.shape[3]])
    Y_all_act = np.empty([0, 4])
    Y_all_weight = np.empty([0, 3])

    # Testing
    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    age_train_labels = age_test_label
    weight_train_labels = weight_test_label

    # activity_index = 0
    # train_data = train_data[act_train_labels[:, activity] == 1]
    # gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
    # age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
    # weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
    # act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_weight_model = load_model("weight_model_DC.hdf5")

    t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    X = t_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)
    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = weight_train_labels
    result1 = eval_weight_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    for activity in range(4):
        print("This is the current activity")
        print(activity)

        # Testing
        train_data = x_test
        act_train_labels = activity_test_label
        gen_train_labels = gender_test_label
        age_train_labels = age_test_label
        weight_train_labels = weight_test_label

        # activity_index = 0
        train_data = train_data[act_train_labels[:, activity] == 1]
        gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
        age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
        weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
        act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        t_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        X = t_data
        Y = act_train_labels
        # print("Activity Identification for Gender 0")
        # print_results(eval_act_model, X, Y)
        # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
        Y = weight_train_labels
        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))
        
        eval_act_model = load_model("activity_model_DC.hdf5")
        eval_weight_model = load_model("weight_model_DC.hdf5")

        pred_act = np.zeros((train_data.shape[0], 4))
        pred_weight = np.zeros((train_data.shape[0], 3))

        X = train_data
        Y_act = eval_act_model.predict(X)
        for index in range(train_data.shape[0]):
            index_act = np.argmax(Y_act[index], axis=0)
            pred_act[index, index_act] = 1
        
        Y_weight = eval_weight_model.predict(X)
        for index in range(train_data.shape[0]):
            gen = np.argmax(Y_weight[index], axis=0)
            if gen == 0:
                pred_weight[index, 0] = 1
            elif gen == 1:
                pred_weight[index, 1] = 1
            else:
                pred_weight[index, 2] = 1
        
        hat_train_data = np.empty((0,768), float)
        hat_wght_data = np.empty((0,3), float)

        for act_inside in range(4):
            print(act_inside)
            Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
            X_inside = train_data[pred_act[:, act_inside] == 1]
            Y_weight_inside = pred_weight[pred_act[:, act_inside] == 1]
            Y_test_wght = weight_train_labels[pred_act[:, act_inside] == 1]
            
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
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                
                for i in range(Y_weight_inside.shape[0]):
                    if Y_weight_inside[i, 0] == 1:
                        y_dataset[i, 0] = 1
                    elif Y_weight_inside[i, 1] == 1:
                        y_dataset[i, 1] = 1
                    elif Y_weight_inside[i, 2] == 1:
                        y_dataset[i, 2] = 1
                
                tensor_Y = torch.from_numpy(y_dataset)
                data_dataset = TensorDataset(tensor_X, tensor_Y)
                train_loader = torch.utils.data.DataLoader(data_dataset, batch_size=256, shuffle=False)

                for batch_idx, (x, y) in enumerate(train_loader):
                    x= Variable(x)
                    y = Variable(y)
                    if(usecuda):
                        x = x.cuda()
                        y = y.cuda()
                    # x_cat = torch.cat((x, y), dim=1)
                    z_e = encodermodel(x)[0]
                    z = np.append(z, z_e.data.cpu(), axis=0)

                z_train = z.copy()
                tensor_z = torch.from_numpy(z_train) # transform to torch tensor
                y_dataset = np.zeros((Y_weight_inside.shape[0], 3))
                
                # rand_2 = secrets.randbelow(100)/100
                # rand_1 = secrets.randbelow(100)/100

                for i in range(Y_weight_inside.shape[0]):
                    rand_1 = secrets.randbelow(10000)+1/10000
                    rand_2 = secrets.randbelow(10000)+1/10000
                    rand_3 = secrets.randbelow(10000)+1/10000
                    sum_numbers = rand_1 + rand_2 + rand_3
                    y_dataset[i, 0] = rand_1/sum_numbers
                    y_dataset[i, 1] = rand_2/sum_numbers
                    y_dataset[i, 2] = rand_3/sum_numbers
                tensor_y = torch.from_numpy(y_dataset)

                z_dataset = TensorDataset(tensor_z, tensor_y)
                z_loader = torch.utils.data.DataLoader(z_dataset, batch_size=256, shuffle=False)

                for batch_idx, (z, y) in enumerate(z_loader):
                    z = Variable(z)
                    y = Variable(y)
                    if(use_gpu):
                        z = z.cuda()
                        y = y.cuda()
                    z_cat = torch.cat((z, y), dim=1)
                    x_hat = decodermodel(z_cat)
                    hat_train_data = np.append(hat_train_data, x_hat.data.cpu(), axis=0)
                hat_wght_data = np.append(hat_wght_data, Y_test_wght, axis=0)

        X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
        Y = act_train_labels
        # print("Activity Identification:")
        # print_results(eval_act_model, X, Y)
        X_all = np.append(X_all, X, axis=0)
        Y_all_act = np.append(Y_all_act, Y, axis=0)

        Y = hat_wght_data
        Y_all_weight = np.append(Y_all_weight, Y, axis=0)

        # result1 = eval_weight_model.evaluate(X, Y)
        # act_acc = round(result1[1], 4) * 100
        # print("Weight Identification: " + str(act_acc))

    result1 = eval_act_model.evaluate(X_all, Y_all_act)
    act_acc = round(result1[1], 4) * 100
    result1 = eval_weight_model.evaluate(X_all, Y_all_weight)
    act_weight = round(result1[1], 4) * 100

    print("Activity Identification " + str(act_acc))
    print("Gender Identification " + str(act_weight))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, Y_all_weight, test_size=0.2, random_state=42)

    eval_weight(X_te, y_te)
    eval_weight_test = load_model("weight_model_test.hdf5")
    result1 = eval_weight_test.evaluate(X_te, y_te)
    act_acc = round(result1[1], 4) * 100
    print("Weight re-Identification " + str(act_acc))
    accuracy_reinference[looooop, 2] = act_acc

# numpy_data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(data=accuracy_reinference, columns=["det", "prob", "randvect"])
df.to_csv('reid_mobi_w.csv', index=False)