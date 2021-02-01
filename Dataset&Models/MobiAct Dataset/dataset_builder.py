act_list = ["WAL", "STD", "JOG", "STU"]
File_path = "/home/omid/pycharm/Mobi/MobiAct_Dataset_v2.0/MobiAct_Dataset_v2.0/Annotated Data"

import os
import numpy as np

a = np.zeros((4, 67))

for i, dir in enumerate(act_list):
    print("###################")
    count = 0
    print(dir)
    for filename in os.listdir(File_path+"/"+dir):
        splits = filename.split("_")
        count = count + 1
        a[i, int(splits[1])-1] = 1

b = np.zeros(67)
for i in range(67):
    b[i] = a[0, i] * a[1, i] * a[2, i] * a[3, i]


import pandas as pd
data_subjects = pd.read_csv("/home/omid/pycharm/Mobi/data_subjects.csv")
print(data_subjects)

sensor_list = ["acc", "gyro"]
dt_list = []
mode = "raw"

for t in sensor_list:
    dt_list.append([t+"_x", t+"_y", t+"_z"])

if mode == "mag":
    dataset = np.zeros((0, len(dt_list)+5))
else:
    dataset = np.zeros((0, len(dt_list)*3+5))
count = 0

for j in data_subjects["id"]:
    i = j - 1
    print(str(count)+"--->"+str(i))
    for act in act_list:
        if b[i] == 1:
            directory = File_path+"/"+act+"/"+act+"_"+str(i+1)+"_1_annotated.csv"
            raw_data = pd.read_csv(directory)
            raw_data = raw_data.drop(["timestamp"], axis=1)
            raw_data = raw_data.drop(["rel_time"], axis=1)

            if mode == "mag":
                values = np.zeros((len(raw_data), len(dt_list)+5))
            else:
                values = np.zeros((len(raw_data), len(dt_list)*3+5))

            for x_id, axes in enumerate(dt_list):
                if mode == "mag":
                    values[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5
                else:
                    values[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
            for iter_i in range(len(raw_data)):
                if raw_data.loc[iter_i, "label"] == "STD":
                    values[iter_i, -1] = 1
                if raw_data.loc[iter_i, "label"] == "WAL":
                    values[iter_i, -1] = 0
                if raw_data.loc[iter_i, "label"] == "JOG":
                    values[iter_i, -1] = 2
                if raw_data.loc[iter_i, "label"] == "STU":
                    values[iter_i, -1] = 3
                if data_subjects.loc[count, "gender"] == "m":
                    values[iter_i, -2] = 0
                if data_subjects.loc[count, "gender"] == "f":
                    values[iter_i, -2] = 1
                values[iter_i, -3] = data_subjects.loc[count, "age"]
                values[iter_i, -4] = j
                values[iter_i, -5] = data_subjects.loc[count, "weight"]
        dataset = np.append(dataset, values, axis=0)
    count = count + 1


datasetcopy = dataset.copy()

cols = []
for axes in dt_list:
    if mode == "raw":
        cols += axes
    else:
        cols += [str(axes[0][:-5])]

if True:
    cols += ["weight", "id", "age", "gender", "act"]

DS = pd.DataFrame(data=datasetcopy, columns=cols)

def ts_to_secs(dataset, w, s, standardize = False, **options):

    data = DS[dataset.columns[:-5]].values
    print(data)
    weight_labels = dataset["weight"].values
    id_labels = dataset["id"].values
    act_labels = dataset["act"].values
    gen_labels = dataset["gender"].values
    age_labels = dataset["age"].values
    print(weight_labels)
    print(id_labels)
    print(gen_labels)
    print(act_labels)
    print(age_labels)

    mean = 0
    std = 1
    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("[INFO] -- Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print("[INFO] -- Training Data has been standardized: the mean is = "+str(mean)+" ; and the std is = "+str(std))

        data -= mean
        data /= std
        print(data)
    else:
        print("[INFO] -- Without Standardization.....")

    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    m = data.shape[0]   # Data Dimension
    ttp = data.shape[1] # Total Time Points
    number_of_secs = int(round(((ttp - w)/s)))

    ##  Create a 3D matrix for Storing Sections
    secs_data = np.zeros((number_of_secs , m , w ))
    weights_secs_labels = np.zeros(number_of_secs)
    id_secs_labels = np.zeros(number_of_secs)
    act_secs_labels = np.zeros(number_of_secs)
    gen_secs_labels = np.zeros(number_of_secs)
    age_secs_labels = np.zeros(number_of_secs)

    k=0
    for i in range(0 , ttp-w, s):
        j = i // s
        if j >= number_of_secs:
            break
        if act_labels[i] != act_labels[i+w-1]:
            continue

        secs_data[k] = data[:, i:i+w]
        weights_secs_labels[k] = weight_labels[i].astype(int)
        id_secs_labels[k] = id_labels[i].astype(int)
        act_secs_labels[k] = act_labels[i].astype(int)
        gen_secs_labels[k] = gen_labels[i].astype(int)
        age_secs_labels[k] = age_labels[i].astype(int)
        k = k+1

    secs_data = secs_data[0:k]
    weights_secs_labels = weights_secs_labels[0:k]
    id_secs_labels = id_secs_labels[0:k]
    act_secs_labels = act_secs_labels[0:k]
    gen_secs_labels = gen_secs_labels[0:k]
    age_secs_labels = age_secs_labels[0:k]
    return secs_data, act_secs_labels, gen_secs_labels, age_secs_labels, id_secs_labels, weights_secs_labels, mean, std

total_data, activity_labels, gender_labels, age_labels, id_labels, weights_labels, mean, std = ts_to_secs(DS, 128, 10, True)

np.save("Data/total_data", total_data)
np.save("Data/weights_data", weights_labels)
np.save("Data/id_labels", id_labels)
np.save("Data/activity_labels", activity_labels)
np.save("Data/gender_labels", gender_labels)
np.save("Data/age_labels", age_labels)

print(age_labels)