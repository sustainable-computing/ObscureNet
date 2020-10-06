import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
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
from torchsummary import summary

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

usecuda = False
use_gpu = False
idgpu = 0
x_dim = 3
AI = 1 #Activity Index
zed = [5]
ma_rate = 0.001

for z_dim in zed:
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(768, 2048)
            self.fc3 = nn.Linear(2048, 1024)
            self.fc4 = nn.Linear(1024, 512)
            self.fc5 = nn.Linear(512, 256)
            self.fc6 = nn.Linear(256, 128)
            self.z_mean = nn.Linear(128, z_dim)
            self.z_log_var = nn.Linear(128, z_dim)
            self.relu = nn.ReLU()

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self,x):
            h1 = self.relu(self.fc1(x))
            h3 = self.relu(self.fc3(h1))
            h4 = self.relu(self.fc4(h3))
            h5 = self.relu(self.fc5(h4))
            h6 = self.relu(self.fc6(h5))
            z_m = self.z_mean(h6)
            z_l = self.z_log_var(h6)
            z = self.reparameterize(z_m, z_l)
            return z, z_m, z_l

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(z_dim, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 512)
            self.fc4 = nn.Linear(512, 1024)
            self.fc5 = nn.Linear(1024, 2048)
            self.fc6 = nn.Linear(2048, 768)
            self.relu = nn.ReLU()

        def forward(self,x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
            h5 = self.relu(self.fc5(h4))
            h6 = self.fc6(h5)
            return h6

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

    for activity in ["all"]:

        # x_vae = x_train[activity_train_label[:, activity] == 1]
        # act_vae = activity_train_label[activity_train_label[:, activity] == 1]
        # weight_vae = weight_train_label[activity_train_label[:, activity] == 1]

        x_vae = x_train
        act_vae = activity_train_label
        weight_vae = weight_train_label

        x_vae_size = x_vae.shape[0]
        x_vae = np.reshape(x_vae, [x_vae_size, 768])
        
        tensor_x = torch.from_numpy(x_vae.astype('float32')) # transform to torch tensor
        tensor_y = torch.from_numpy(weight_vae.astype('float32'))
        vae_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = torch.utils.data.DataLoader(vae_dataset, batch_size=512, shuffle=True)

        encodermodel = Encoder()
        if usecuda:
            encodermodel.cuda(idgpu)
        decodermodel = Decoder()
        if usecuda:
            decodermodel.cuda(idgpu)

        optimizerencoder = optim.Adam(encodermodel.parameters())
        optimizerdecoder = optim.Adam(decodermodel.parameters())
'''
        for i in range(100):
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
                
                train_z, mu, log_var = encodermodel(train_x)
                train_xr = decodermodel(train_z)

                recons_loss = F.mse_loss(train_xr, train_x)*512
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = torch.mean((recons_loss + kld_loss)/150)
                loss.backward()

                optimizerencoder.step()
                optimizerdecoder.step()

                if(batch_idx%100 == 0):
                    print("Epoch %d : MSE is %f, KLD loss is %f" % (i,recons_loss.data, kld_loss.data))
        torch.save(encodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/single_vae_encoder_'+activity+str(z_dim))
        torch.save(decodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/single_vae_decoder_'+activity+str(z_dim))
'''
z_dim = 5
def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

latent_means = np.zeros((5, 3, z_dim))

eval_act_model = load_model("activity_model_DC.hdf5")
eval_weight_model = load_model("weight_model_DC.hdf5")
'''
encodermodel = Encoder().double()
encodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/single_vae_encoder_all'+str(z_dim)))
if usecuda:
    encodermodel.cuda(idgpu)

decodermodel = Decoder().double()
decodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/single_vae_decoder_all'+str(z_dim)))
if usecuda:
    decodermodel.cuda(idgpu)

for activity in range(4):
    train_data = x_train
    act_train_labels = activity_train_label
    gen_train_labels = gender_train_label
    age_train_labels = age_train_label
    weight_train_labels = weight_train_label

    # activity_index = 0
    train_data = train_data[act_train_labels[:, activity] == 1]
    gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
    age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
    weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
    act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

    # # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    # X = train_data
    # Y = act_train_labels
    # print("Activity Identification for Gender 0")
    # print_results(eval_act_model, X, Y)

    # # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    # Y = weight_train_labels
    # result1 = eval_weight_model.evaluate(X, Y)
    # act_acc = round(result1[1], 4) * 100
    # print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

    ### Manipulation at the Age Level
    train_data_0 = train_data[weight_train_labels[:, 0] == 1]
    act_train_labels_0 = act_train_labels[weight_train_labels[:, 0] == 1]
    train_data_1 = train_data[weight_train_labels[:, 1] == 1]
    act_train_labels_1 = act_train_labels[weight_train_labels[:, 1] == 1]
    train_data_2 = train_data[weight_train_labels[:, 2] == 1]
    act_train_labels_2 = act_train_labels[weight_train_labels[:, 2] == 1]
    weight_train_data_0 = np.zeros((train_data_0.shape[0], 3))
    weight_train_data_1 = np.zeros((train_data_1.shape[0], 3))
    weight_train_data_2 = np.zeros((train_data_2.shape[0], 3))

    for j in range(weight_train_data_0.shape[0]):
        weight_train_data_0[j, 0] = 1
    for j in range(weight_train_data_1.shape[0]):
        weight_train_data_1[j, 1] = 1
    for j in range(weight_train_data_2.shape[0]):
        weight_train_data_2[j, 2] = 1

    train_data_0 = np.reshape(train_data_0, [train_data_0.shape[0], 768])
    train_data_1 = np.reshape(train_data_1, [train_data_1.shape[0], 768])
    train_data_2 = np.reshape(train_data_2, [train_data_2.shape[0], 768])

    tensor_data_0 = torch.from_numpy(train_data_0) # transform to torch tensor
    y_0_dataset = np.zeros((weight_train_data_0.shape[0], 3))
    for i in range(weight_train_data_0.shape[0]):
        y_0_dataset[i, 0] = 1
    tensor_y_0 = torch.from_numpy(y_0_dataset)
    data_0_dataset = TensorDataset(tensor_data_0, tensor_y_0)

    tensor_data_1 = torch.from_numpy(train_data_1) # transform to torch tensor
    y_1_dataset = np.zeros((weight_train_data_1.shape[0], 3))
    for i in range(weight_train_data_1.shape[0]):
        y_1_dataset[i, 1] = 1
    tensor_y_1 = torch.from_numpy(y_1_dataset)
    data_1_dataset = TensorDataset(tensor_data_1, tensor_y_1)

    tensor_data_2 = torch.from_numpy(train_data_2) # transform to torch tensor
    y_2_dataset = np.zeros((weight_train_data_2.shape[0], 3))
    for i in range(weight_train_data_2.shape[0]):
        y_2_dataset[i, 2] = 1
    tensor_y_2 = torch.from_numpy(y_2_dataset)
    data_2_dataset = TensorDataset(tensor_data_2, tensor_y_2)

    train_0_loader = torch.utils.data.DataLoader(data_0_dataset, batch_size=256, shuffle=False)
    train_1_loader = torch.utils.data.DataLoader(data_1_dataset, batch_size=256, shuffle=False)
    train_2_loader = torch.utils.data.DataLoader(data_2_dataset, batch_size=256, shuffle=False)

    z_0 = np.empty((0,z_dim), float)
    z_1 = np.empty((0,z_dim), float)
    z_2 = np.empty((0,z_dim), float)

    for batch_idx, (x, y) in enumerate(train_0_loader):
        x= Variable(x)
        y = Variable(y)
        if(usecuda):
            x = x.cuda(idgpu)
            y = y.cuda(idgpu)
        z_e = encodermodel(x)[0]
        z_0 = np.append(z_0, z_e.data.cpu(), axis=0)

    for batch_idx, (x, y) in enumerate(train_1_loader):
        x= Variable(x)
        y = Variable(y)
        if(usecuda):
            x = x.cuda(idgpu)
            y = y.cuda(idgpu)
        z_e = encodermodel(x)[0]
        z_1 = np.append(z_1, z_e.data.cpu(), axis=0)
    
    for batch_idx, (x, y) in enumerate(train_2_loader):
        x= Variable(x)
        y = Variable(y)
        if(usecuda):
            x = x.cuda(idgpu)
            y = y.cuda(idgpu)
        z_e = encodermodel(x)[0]
        z_2 = np.append(z_2, z_e.data.cpu(), axis=0)
    
    z_train_0 = z_0
    z_train_1 = z_1
    z_train_2 = z_2

    latent_means[activity, 0, :] = np.mean(z_train_0, axis=0)
    latent_means[activity, 1, :] = np.mean(z_train_1, axis=0)
    latent_means[activity, 2, :] = np.mean(z_train_2, axis=0)
np.save("/home/omid/pycharm/Mobi/latent", latent_means)
'''
# for activity in range(4):
#     encodermodel = Encoder().double()
#     encodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/multi_vae_encoder_weight_'+str(activity)+str(z_dim)))
#     if usecuda:
#         encodermodel.cuda(idgpu)

#     decodermodel = Decoder().double()
#     decodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/multi_vae_decoder_weight_'+str(activity)+str(z_dim)))
#     if usecuda:
#         decodermodel.cuda(idgpu)
    
#     train_data = x_test
#     act_train_labels = activity_test_label
#     gen_train_labels = gender_test_label
#     age_train_labels = age_test_label
#     weight_train_labels = weight_test_label

#     # activity_index = 0
#     train_data = train_data[act_train_labels[:, activity] == 1]
#     gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
#     age_train_labels = age_train_labels[act_train_labels[:, activity] == 1]
#     weight_train_labels = weight_train_labels[act_train_labels[:, activity] == 1]
#     act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]

#     eval_act_model = load_model("activity_model_DC.hdf5")
#     eval_weight_model = load_model("weight_model_DC.hdf5")

#     # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
#     X = train_data
#     Y = act_train_labels
#     print("Activity Identification for Gender 0")
#     print_results(eval_act_model, X, Y)

#     # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
#     Y = weight_train_labels
#     result1 = eval_weight_model.evaluate(X, Y)
#     act_acc = round(result1[1], 4) * 100
#     print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

#     ### Manipulation at the Age Level
#     train_data_0 = train_data[weight_train_labels[:, 0] == 1]
#     act_train_labels_0 = act_train_labels[weight_train_labels[:, 0] == 1]
#     train_data_1 = train_data[weight_train_labels[:, 1] == 1]
#     act_train_labels_1 = act_train_labels[weight_train_labels[:, 1] == 1]
#     train_data_2 = train_data[weight_train_labels[:, 2] == 1]
#     act_train_labels_2 = act_train_labels[weight_train_labels[:, 2] == 1]
#     weight_train_data_0 = np.zeros((train_data_0.shape[0], 3))
#     weight_train_data_1 = np.zeros((train_data_1.shape[0], 3))
#     weight_train_data_2 = np.zeros((train_data_2.shape[0], 3))

#     for j in range(weight_train_data_0.shape[0]):
#         weight_train_data_0[j, 0] = 1
#     for j in range(weight_train_data_1.shape[0]):
#         weight_train_data_1[j, 1] = 1
#     for j in range(weight_train_data_2.shape[0]):
#         weight_train_data_2[j, 2] = 1

#     eval_act_model = load_model("activity_model_DC.hdf5")
#     eval_weight_model = load_model("weight_model_DC.hdf5")

#     train_data_0 = np.reshape(train_data_0, [train_data_0.shape[0], 768])
#     train_data_1 = np.reshape(train_data_1, [train_data_1.shape[0], 768])
#     train_data_2 = np.reshape(train_data_2, [train_data_2.shape[0], 768])

#     tensor_data_0 = torch.from_numpy(train_data_0) # transform to torch tensor
#     y_0_dataset = np.zeros((weight_train_data_0.shape[0], 3))
#     for i in range(weight_train_data_0.shape[0]):
#         y_0_dataset[i, 0] = 1
#     tensor_y_0 = torch.from_numpy(y_0_dataset)
#     data_0_dataset = TensorDataset(tensor_data_0, tensor_y_0)

#     tensor_data_1 = torch.from_numpy(train_data_1) # transform to torch tensor
#     y_1_dataset = np.zeros((weight_train_data_1.shape[0], 3))
#     for i in range(weight_train_data_1.shape[0]):
#         y_1_dataset[i, 1] = 1
#     tensor_y_1 = torch.from_numpy(y_1_dataset)
#     data_1_dataset = TensorDataset(tensor_data_1, tensor_y_1)

#     tensor_data_2 = torch.from_numpy(train_data_2) # transform to torch tensor
#     y_2_dataset = np.zeros((weight_train_data_2.shape[0], 3))
#     for i in range(weight_train_data_2.shape[0]):
#         y_2_dataset[i, 2] = 1
#     tensor_y_2 = torch.from_numpy(y_2_dataset)
#     data_2_dataset = TensorDataset(tensor_data_2, tensor_y_2)

#     train_0_loader = torch.utils.data.DataLoader(data_0_dataset, batch_size=256, shuffle=False)
#     train_1_loader = torch.utils.data.DataLoader(data_1_dataset, batch_size=256, shuffle=False)
#     train_2_loader = torch.utils.data.DataLoader(data_2_dataset, batch_size=256, shuffle=False)

#     z_0 = np.empty((0,z_dim), float)
#     z_1 = np.empty((0,z_dim), float)
#     z_2 = np.empty((0,z_dim), float)

#     for batch_idx, (x, y) in enumerate(train_0_loader):
#         x= Variable(x)
#         y = Variable(y)
#         if(usecuda):
#             x = x.cuda(idgpu)
#             y = y.cuda(idgpu)
#         z_e = encodermodel(x)[0]
#         z_0 = np.append(z_0, z_e.data.cpu(), axis=0)

#     for batch_idx, (x, y) in enumerate(train_1_loader):
#         x= Variable(x)
#         y = Variable(y)
#         if(usecuda):
#             x = x.cuda(idgpu)
#             y = y.cuda(idgpu)
#         z_e = encodermodel(x)[0]
#         z_1 = np.append(z_1, z_e.data.cpu(), axis=0)
    
#     for batch_idx, (x, y) in enumerate(train_2_loader):
#         x= Variable(x)
#         y = Variable(y)
#         if(usecuda):
#             x = x.cuda(idgpu)
#             y = y.cuda(idgpu)
#         z_e = encodermodel(x)[0]
#         z_2 = np.append(z_2, z_e.data.cpu(), axis=0)
    
#     z_train_0 = z_0
#     z_train_1 = z_1
#     z_train_2 = z_2

#     z_train_0 = z_train_0 - latent_means[activity, 0, :] + latent_means[activity, 2, :]
#     z_train_1 = z_train_1 - latent_means[activity, 1, :] + latent_means[activity, 0, :]
#     z_train_2 = z_train_2 - latent_means[activity, 2, :] + latent_means[activity, 1, :]

#     tensor_z_0 = torch.from_numpy(z_train_0) # transform to torch tensor
#     y_0_dataset = np.zeros((weight_train_data_0.shape[0], 3))
#     for i in range(weight_train_data_0.shape[0]):
#         y_0_dataset[i, 2] = 1
#     tensor_y_0 = torch.from_numpy(y_0_dataset)

#     tensor_z_1 = torch.from_numpy(z_train_1) # transform to torch tensor
#     y_1_dataset = np.zeros((weight_train_data_1.shape[0], 3))
#     for i in range(weight_train_data_1.shape[0]):
#         y_1_dataset[i, 0] = 1
#     tensor_y_1 = torch.from_numpy(y_1_dataset)

#     tensor_z_2 = torch.from_numpy(z_train_2) # transform to torch tensor
#     y_2_dataset = np.zeros((weight_train_data_2.shape[0], 3))
#     for i in range(weight_train_data_2.shape[0]):
#         y_2_dataset[i, 1] = 1
#     tensor_y_2 = torch.from_numpy(y_2_dataset)

#     z_0_dataset = TensorDataset(tensor_z_0, tensor_y_0)
#     z_1_dataset = TensorDataset(tensor_z_1, tensor_y_1)
#     z_2_dataset = TensorDataset(tensor_z_2, tensor_y_2)

#     z_0_loader = torch.utils.data.DataLoader(z_0_dataset, batch_size=256, shuffle=False)
#     z_1_loader = torch.utils.data.DataLoader(z_1_dataset, batch_size=256, shuffle=False)
#     z_2_loader = torch.utils.data.DataLoader(z_2_dataset, batch_size=256, shuffle=False)

#     hat_train_data_0 = np.empty((0,768), float)
#     hat_train_data_1 = np.empty((0,768), float)
#     hat_train_data_2 = np.empty((0,768), float)

#     for batch_idx, (z, y) in enumerate(z_0_loader):
#         z = Variable(z)
#         y = Variable(y)
#         if(use_gpu):
#             z = z.cuda(idgpu)
#             y = y.cuda(idgpu)
#         x_hat = decodermodel(z)
#         hat_train_data_0 = np.append(hat_train_data_0, x_hat.data.cpu(), axis=0)

#     for batch_idx, (z, y) in enumerate(z_1_loader):
#         z = Variable(z)
#         y = Variable(y)
#         if(use_gpu):
#             z = z.cuda(idgpu)
#             y = y.cuda(idgpu)
#         x_hat = decodermodel(z)
#         hat_train_data_1 = np.append(hat_train_data_1, x_hat.data.cpu(), axis=0)

#     for batch_idx, (z, y) in enumerate(z_2_loader):
#         z = Variable(z)
#         y = Variable(y)
#         if(use_gpu):
#             z = z.cuda(idgpu)
#             y = y.cuda(idgpu)
#         x_hat = decodermodel(z)
#         hat_train_data_2 = np.append(hat_train_data_2, x_hat.data.cpu(), axis=0)

#     hat_train_data = np.concatenate((hat_train_data_0, hat_train_data_1, hat_train_data_2), axis=0)
#     hat_act_train_labels = np.concatenate((act_train_labels_0, act_train_labels_1, act_train_labels_2), axis=0)
#     hat_weight_train_labels = np.concatenate((weight_train_data_0, weight_train_data_1, weight_train_data_2), axis=0)

#     eval_act_model = load_model("activity_model_DC.hdf5")
#     eval_weight_model = load_model("weight_model_DC.hdf5")

#     # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
#     reconstructed_input = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
#     X = reconstructed_input
#     Y = hat_act_train_labels
#     print("Activity Identification for Gender 0")
#     print_results(eval_act_model, X, Y)

#     # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
#     Y = hat_weight_train_labels
#     result1 = eval_weight_model.evaluate(X, Y)
#     act_acc = round(result1[1], 4) * 100
#     print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))

latent_means = np.load("/home/omid/pycharm/Mobi/latent.npy")

act_label = 0
gen_label = 0
# data_subjects = pd.read_csv("/home/omid/pycharm/Mobi/data_subjects.csv")
# data = np.load("/home/omid/pycharm/Mobi/Data/total_data.npy", allow_pickle=True)
# activity = np.load("/home/omid/pycharm/Mobi/Data/activity_labels.npy", allow_pickle=True)
# gender = np.load("/home/omid/pycharm/Mobi/Data/gender_labels.npy", allow_pickle=True)
# age = np.load("/home/omid/pycharm/Mobi/Data/age_labels.npy", allow_pickle=True)
# id = np.load("/home/omid/pycharm/Mobi/Data/id_labels.npy", allow_pickle=True)

# array = np.arange(data.shape[0])
# np.random.shuffle(array)
# data = data[array]
# activity = activity[array]
# gender = gender[array]
# age = age[array]
# id = id[array]

# data_train = np.array([]).reshape(0, data.shape[1], data.shape[2])
# data_test = np.array([]).reshape(0, data.shape[1], data.shape[2])
# activity_train = np.array([])
# activity_test = np.array([])
# age_train = np.array([])
# age_test = np.array([])
# gender_train = np.array([])
# gender_test = np.array([])

# for i in data_subjects["id"]:
#     data_sub_id = data[id[:] == i]
#     age_sub_id = age[id[:] == i]
#     activity_sub_id = activity[id[:] == i]
#     gender_sub_id = gender[id[:] == i]
#     x_train, x_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(data_sub_id, age_sub_id, activity_sub_id, gender_sub_id, test_size = 0.2, random_state = 42)
#     data_train = np.concatenate((data_train, x_train), axis=0)
#     data_test = np.concatenate((data_test, x_test), axis=0)
#     age_train = np.concatenate((age_train, y_train), axis=0)
#     age_test = np.concatenate((age_test, y_test), axis=0)
#     activity_train = np.concatenate((activity_train, z_train), axis=0)
#     activity_test = np.concatenate((activity_test, z_test), axis=0)
#     gender_train = np.concatenate((gender_train, w_train), axis=0)
#     gender_test = np.concatenate((gender_test, w_test), axis=0)

# nb_classes = len(np.unique(activity_train[:]))
# activity_train_label = keras.utils.to_categorical(activity_train[:], nb_classes)
# nb_classes = len(np.unique(activity_test[:]))
# activity_test_label = keras.utils.to_categorical(activity_test[:], nb_classes)
# nb_classes = len(np.unique(age_train[:]))
# age_train_label = keras.utils.to_categorical(age_train[:], nb_classes)
# nb_classes = len(np.unique(age_test[:]))
# age_test_label = keras.utils.to_categorical(age_test[:], nb_classes)
# gender_train_label = gender_train
# gender_test_label = gender_test
# x_train = data_train.reshape((data_train.shape[0], data_train.shape[1], data_train.shape[2], 1))
# x_test = data_test.reshape((data_test.shape[0], data_test.shape[1], data_test.shape[2], 1))

encodermodel_0 = Encoder().double()
encodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/single_vae_encoder_all'+str(z_dim)))
if usecuda:
    encodermodel_0.cuda(idgpu)
decodermodel_0 = Decoder().double()
decodermodel_0.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/single_vae_decoder_all'+str(z_dim)))
if usecuda:
    decodermodel_0.cuda(idgpu)

print(encodermodel_0)
print(decodermodel_0)

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

    X = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]))
    # X = train_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = weight_train_labels
    result1 = eval_weight_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: Weight Train Accuracy Gender 0: " + str(act_acc))
    
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
    
    for act_inside in range(4):
        print(act_inside)
        Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
        X_inside = train_data[pred_act[:, act_inside] == 1]
        Y_weight_inside = pred_weight[pred_act[:, act_inside] == 1]
        
        if Y_act_inside != []:
            encodermodel = encodermodel_0
            decodermodel = decodermodel_0

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
                    x = x.cuda(idgpu)
                    y = y.cuda(idgpu)
                z_e = encodermodel(x)[0]
                z = np.append(z, z_e.data.cpu(), axis=0)

            z_train = z.copy()

            for l in range(z_train.shape[0]):
                if Y_weight_inside[l, 0] == 1:
                    z_train[l] = z_train[l] - latent_means[activity, 0, :] + latent_means[activity, 2, :]
                elif Y_weight_inside[1, 1] == 1:
                    z_train[l] = z_train[l] - latent_means[activity, 1, :] + latent_means[activity, 0, :]
                else:
                    z_train[l] = z_train[l] - latent_means[activity, 2, :] + latent_means[activity, 1, :]
            
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
                    z = z.cuda(idgpu)
                    y = y.cuda(idgpu)
                x_hat = decodermodel(z)
                hat_train_data = np.append(hat_train_data, x_hat.data.cpu(), axis=0)
    
    X = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    Y = act_train_labels
    print("Activity Identification:")
    print_results(eval_act_model, X, Y)

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = weight_train_labels
    result1 = eval_weight_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("Weight Identification: " + str(act_acc))
