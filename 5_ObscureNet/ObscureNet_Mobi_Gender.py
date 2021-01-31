import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";
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
x_dim = 2
AI = 0 #Activity Index
zed = [5]
ma_rate = 0.001

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
        print(x_vae.shape)
        x_vae_size = x_vae.shape[0]
        x_vae = np.reshape(x_vae, [x_vae_size, 768])
        print(x_vae.shape)
        print(act_vae.shape)
        print(gen_vae.shape)
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

                # cat_x = torch.cat((train_x, train_y), dim = 1)
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
                kld_loss = torch.mean(-2 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = torch.mean((recons_loss + kld_loss)/150) + 0.2*vaeLoss
                # loss = torch.mean((recons_loss + kld_loss)/150)
                loss.backward()

                # optimizer_aux.zero_grad()
                optimizerencoder.step()
                optimizerdecoder.step()

                if(batch_idx%100 == 0):
                    # result1 = aux(train_z)
                    # _, result1 = torch.max(result1.data, 1)
                    # correct = (result1 == y).float().sum()
                    # accuracy = 100 * correct / (train_z.shape[0])
                    # print("***[RESULT]*** Aux results" + str(accuracy.data))
                    print("Epoch %d : MSE is %f, KLD loss is %f, AUX loss is %f" % (i,recons_loss.data, kld_loss.data, auxLoss))

        torch.save(encodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_g_encoder_alpha_02_beta_2_'+str(activity)+str(z_dim))
        torch.save(decodermodel.state_dict(), '/home/omid/pycharm/Mobi/models/obs_mobi_g_decoder_alpha_02_beta_2_'+str(activity)+str(z_dim))

'''
z_dim = 5
def print_results(M, X, Y):
    result1 = M.evaluate(X, Y, verbose=2)
    print(result1)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]*** ACT Accuracy: " + str(act_acc))

Latent_means = np.zeros((5, 2, z_dim))
for activity in [1]:
    encodermodel = Encoder().double()
    encodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/newvae_encoder_'+str(activity)+str(z_dim)))
    if usecuda:
        encodermodel.cuda(idgpu)

    decodermodel = Decoder().double()
    decodermodel.load_state_dict(torch.load('/home/omid/pycharm/Mobi/models/newvae_decoder_'+str(activity)+str(z_dim)))
    if usecuda:
        decodermodel.cuda(idgpu)

    # Testing
    train_data = x_test
    act_train_labels = activity_test_label
    gen_train_labels = gender_test_label
    # activity_index = 0
    train_data = train_data[act_train_labels[:, activity] == 1]
    gen_train_labels = gen_train_labels[act_train_labels[:, activity] == 1]
    act_train_labels = act_train_labels[act_train_labels[:, activity] == 1]
    ### Manipulation at the Gender Level
    train_data_0 = train_data[gen_train_labels == 0]
    act_train_labels_0 = act_train_labels[gen_train_labels == 0]
    train_data_1 = train_data[gen_train_labels == 1]
    act_train_labels_1 = act_train_labels[gen_train_labels == 1]
    gender_train_data_0 = np.zeros((train_data_0.shape[0]))
    gender_train_data_1 = np.ones((train_data_1.shape[0]))

    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_gen_model = load_model("gender_model_DC.hdf5")

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    X = train_data
    Y = act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: GEN Train Accuracy Gender 0: " + str(act_acc))

    train_data_0 = np.reshape(train_data_0, [train_data_0.shape[0], 768])
    train_data_1 = np.reshape(train_data_1, [train_data_1.shape[0], 768])

    tensor_data_0 = torch.from_numpy(train_data_0) # transform to torch tensor
    y_0_dataset = np.zeros((gender_train_data_0.shape[0], 2))
    for i in range(gender_train_data_0.shape[0]):
        y_0_dataset[i, 0] = 1
    tensor_y_0 = torch.from_numpy(y_0_dataset)

    tensor_data_1 = torch.from_numpy(train_data_1) # transform to torch tensor
    y_1_dataset = np.zeros((gender_train_data_1.shape[0], 2))
    for i in range(gender_train_data_1.shape[0]):
        y_1_dataset[i, 1] = 1
    tensor_y_1 = torch.from_numpy(y_1_dataset)

    data_0_dataset = TensorDataset(tensor_data_0, tensor_y_0)
    data_1_dataset = TensorDataset(tensor_data_1, tensor_y_1)

    train_0_loader = torch.utils.data.DataLoader(data_0_dataset, batch_size=256, shuffle=False)
    train_1_loader = torch.utils.data.DataLoader(data_1_dataset, batch_size=256, shuffle=False)

    # tensor_data_0 = torch.from_numpy(train_data_0) # transform to torch tensor
    # data_0_dataset = TensorDataset(tensor_data_0)
    # tensor_data_1 = torch.from_numpy(train_data_1) # transform to torch tensor
    # data_1_dataset = TensorDataset(tensor_data_1)
    # train_0_loader = torch.utils.data.DataLoader(data_0_dataset, batch_size=256, shuffle=False)
    # train_1_loader = torch.utils.data.DataLoader(data_1_dataset, batch_size=256, shuffle=False)

    z_0 = np.empty((0,z_dim), float)
    z_1 = np.empty((0,z_dim), float)

    for batch_idx, (train_x, train_y) in enumerate(train_0_loader):
        train_x= Variable(train_x)
        train_y= Variable(train_y)
        if(usecuda):
            train_x = train_x.cuda(idgpu)
            train_y = train_y.cuda(idgpu)
        # x_cat = torch.cat((train_x, train_y), dim=1)
        z = encodermodel(train_x)[0]
        z_0 = np.append(z_0, z.data.cpu(), axis=0)

    for batch_idx, (train_x, train_y) in enumerate(train_1_loader):
        train_x= Variable(train_x)
        train_y= Variable(train_y)
        if(usecuda):
            train_x = train_x.cuda(idgpu)
            train_y = train_y.cuda(idgpu)
        # x_cat = torch.cat((train_x, train_y), dim=1)
        z = encodermodel(train_x)[0]
        z_1 = np.append(z_1, z.data.cpu(), axis=0)

    z_train_0 = z_0
    z_train_1 = z_1

    # mean_0 = np.mean(z_train_0, axis=0)
    # mean_1 = np.mean(z_train_1, axis=0)
    # z_train_0 = z_train_0 - mean_0 + mean_1
    # z_train_1 = z_train_1 - mean_1 + mean_0

    tensor_z_0 = torch.from_numpy(z_train_0) # transform to torch tensor
    y_0_dataset = np.zeros((gender_train_data_0.shape[0], 2))
    for i in range(gender_train_data_0.shape[0]):
        y_0_dataset[i, 1] = 1
    tensor_y_0 = torch.from_numpy(y_0_dataset)

    tensor_z_1 = torch.from_numpy(z_train_1) # transform to torch tensor
    y_1_dataset = np.zeros((gender_train_data_1.shape[0], 2))
    for i in range(gender_train_data_1.shape[0]):
        y_1_dataset[i, 0] = 1
    tensor_y_1 = torch.from_numpy(y_1_dataset)

    z_0_dataset = TensorDataset(tensor_z_0, tensor_y_0)
    z_1_dataset = TensorDataset(tensor_z_1, tensor_y_1)

    z_0_loader = torch.utils.data.DataLoader(z_0_dataset, batch_size=256, shuffle=False)
    z_1_loader = torch.utils.data.DataLoader(z_1_dataset, batch_size=256, shuffle=False)

    hat_train_data_0 = np.empty((0,768), float)
    hat_train_data_1 = np.empty((0,768), float)

    for batch_idx, (z, y) in enumerate(z_0_loader):
        z = Variable(z)
        y = Variable(y)
        if(use_gpu):
            z = z.cuda(idgpu)
            y = y.cuda(idgpu)
        z_cat = torch.cat((z, y), dim=1)
        x_hat = decodermodel(z_cat)
        hat_train_data_0 = np.append(hat_train_data_0, x_hat.data.cpu(), axis=0)

    for batch_idx, (z, y) in enumerate(z_1_loader):
        z = Variable(z)
        y = Variable(y)
        if(use_gpu):
            z = z.cuda(idgpu)
            y = y.cuda(idgpu)
        z_cat = torch.cat((z, y), dim=1)
        x_hat = decodermodel(z_cat)
        hat_train_data_1 = np.append(hat_train_data_1, x_hat.data.cpu(), axis=0)

    hat_train_data = np.concatenate((hat_train_data_0, hat_train_data_1), axis=0)
    hat_act_train_labels = np.concatenate((act_train_labels_0, act_train_labels_1), axis=0)
    hat_gen_train_labels = np.concatenate((gender_train_data_0, gender_train_data_1), axis=0)

    eval_act_model = load_model("activity_model_DC.hdf5")
    eval_gen_model = load_model("gender_model_DC.hdf5")

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    reconstructed_input = np.reshape(hat_train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2],train_data.shape[3]])
    X = reconstructed_input
    Y = hat_act_train_labels
    print("Activity Identification for Gender 0")
    print_results(eval_act_model, X, Y)

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = hat_gen_train_labels
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("***[RESULT]***Original: GEN Train Accuracy Gender 0: " + str(act_acc))
'''
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
X_all = np.empty([0, x_train.shape[1], x_train.shape[2],x_train.shape[3]])
Y_all_act = np.empty([0, 4])
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

    ### Manipulation at the Gender Level
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
        print(act_inside)
        Y_act_inside = pred_act[pred_act[:, act_inside] == 1]
        X_inside = train_data[pred_act[:, act_inside] == 1]
        Y_gen_inside = pred_gen[pred_act[:, act_inside] == 1]
        Y_test_gen = gen_train_labels[pred_act[:, act_inside] == 1]
        print(Y_act_inside.shape)
        
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
            print(count)
            count = 0
            for i in range(Y_gen_inside.shape[0]):
                if Y_gen_inside[i, 0] == 1:
                    count = count + 1
                    y_dataset[i, 0] = 1
                else:
                    y_dataset[i, 1] = 1
            print(count)
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
                # x_cat = torch.cat((train_x, train_y), dim=1)
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
    print("Activity Identification:")
    print_results(eval_act_model, X, Y)
    X_all = np.append(X_all, X, axis=0)
    Y_all_act = np.append(Y_all_act, Y, axis=0)

    # X = np.reshape(hat_train_data, (hat_train_data.shape[0], 2, 128, 1))
    Y = hat_gen_data
    result1 = eval_gen_model.evaluate(X, Y)
    act_acc = round(result1[1], 4) * 100
    print("Gender Identification: " + str(act_acc))
    Y_all_gen = np.append(Y_all_gen, Y, axis=0)

# result1 = eval_act_model.evaluate(X_all, Y_all_act)
print_act_results_f1_score(eval_act_model, X_all, Y_all_act)
# result1 = eval_gen_model.evaluate(X_all, Y_all_gen)
print_gen_results_f1_score(eval_gen_model, X_all, Y_all_gen)