import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_signals(rnd_smpl, pts, org, tra):
    plt.rcParams['figure.figsize'] = (60,30)
    plt.rcParams['font.size'] = 40
    plt.rcParams['image.cmap'] = 'plasma'
    plt.rcParams['axes.linewidth'] = 2

    dt = ["rotRaw","rotTra","acclRaw","acclTra"]
    clr = ["rp-", "gs-", "bo-", "k*-", "c.-"]   
    lbl = ["DWS", "UPS","WLK", "JOG", "STD"]


    fig, ax = plt.subplots(4, 5, sharex=True, sharey=True)
    for k in range(5):
        raw_rot = org[act_test_labels[:,k]==1.0][rnd_smpl,0,:,0]   
        tra_rot = tra[act_test_labels[:,k]==1.0][rnd_smpl,0,:,0]
        raw_accl = org[act_test_labels[:,k]==1.0][rnd_smpl,1,:,0]   
        tra_accl =tra[act_test_labels[:,k]==1.0][rnd_smpl,1,:,0]
        ax[0,k].plot(np.arange(0.,128./50.,1./50), raw_rot, clr[k], linewidth=5, markersize=5)
        ax[1,k].plot(np.arange(0.,128./50.,1./50), tra_rot, clr[k], linewidth=5, markersize=5)
        ax[2,k].plot(np.arange(0.,128./50.,1./50), raw_accl, clr[k], linewidth=5, markersize=5)
        ax[3,k].plot(np.arange(0.,128./50.,1./50), tra_accl, clr[k], linewidth=5, markersize=5)    
        ax[0,k].set_title(lbl[k])
        if k < 4:
            ax[k,0].set_ylabel(dt[k])


    #plt.setp(ax, yticks=np.arange(-2, 6, 2))
    fig.text(0.5, 0.04, 'second', ha='center')
    #fig.text(0.04, 0.5, 'magnitude value', va='center', rotation='vertical', fontsize=26)
    # ax[0].legend(loc="upper center", fontsize=26)
    plt.save("original_det.pdf")

original = np.load("/home/omid/pycharm/Mobi/data/mobi_g_org.npy")
det = np.load("/home/omid/pycharm/Mobi/data/mobi_g_det.npy")
prob = np.load("/home/omid/pycharm/Mobi/data/mobi_g_prob.npy")
rand = np.load("/home/omid/pycharm/Mobi/data/mobi_g_rand.npy")

rnd_smpl= np.random.randint(100)
pts = 128
print("Random Test Sample: #"+str(rnd_smpl))
plot_signals(rnd_smpl, pts, original, det)