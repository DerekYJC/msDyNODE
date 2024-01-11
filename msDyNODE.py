# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 2023

    Main code for manuscript entitled "Multiscale effective connectivity analysis of 
    brain activity using neural ordinary differential equations"

@author: Yin-Jui Chang (DerekYJC) @ SantacruzLab
"""

#%% Import the required package
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

#%% Define the class/functions for later usage
def spike2fr(spikes, sigma=0.05):
    xx = np.linspace(-2, 2, 401)
    yy = np.exp(-np.square(xx)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    return np.convolve(spikes, yy)[200:-200]

def get_batch(batch_time=500, batch_size=20):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    # s: randomly select the start time during the trajectory
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
class ODEFunc(nn.Module):
    def __init__(self, input_size):
        super(ODEFunc, self).__init__()
        
        self.input_size = input_size
        # Firing rate model -- initialize parameters
        self.i2o    = torch.nn.Parameter(torch.Tensor(input_size, input_size))
        self.o2o    = torch.nn.Parameter(torch.Tensor(input_size, input_size))
        self.tau    = torch.nn.Parameter(torch.Tensor(input_size, ))       
        self.fc = nn.Linear(input_size, input_size)        
        self.mu = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.slope = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.frmax = torch.nn.Parameter(torch.Tensor(input_size, ))        
        nn.init.uniform_(self.i2o,  -10, 10)
        nn.init.uniform_(self.o2o,  -10, 10)
        nn.init.uniform_(self.tau,   10, 50)
        nn.init.uniform_(self.mu,  -5, 5)
        nn.init.uniform_(self.slope,  0.01, 0.1)
        nn.init.uniform_(self.frmax,  5, 15)
        # LFP model -- initialize parameters
        self.A  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.a  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.B  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.b  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.C1 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.C2 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.C3 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.C4 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.p_mu = torch.nn.Parameter(torch.Tensor(input_size, ))          
        self.fr_max_0     = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.sigm_slope_0 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.threshold_0  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.fr_max_1     = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.sigm_slope_1 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.threshold_1  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.fr_max_2     = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.sigm_slope_2 = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.threshold_2  = torch.nn.Parameter(torch.Tensor(input_size, ))
        self.lfp2lfp_C   = torch.nn.Parameter(torch.Tensor(input_size, input_size))        
        nn.init.uniform_(self.A, 2, 14)
        nn.init.uniform_(self.a, 90, 100)
        nn.init.uniform_(self.B, 1, 10)
        nn.init.uniform_(self.b, 40, 60)
        nn.init.uniform_(self.C1, 130, 140)
        nn.init.uniform_(self.C2, 104, 112)
        nn.init.uniform_(self.C3, 32, 35)
        nn.init.uniform_(self.C4, 32, 35)
        nn.init.uniform_(self.p_mu, 10, 150) 
        nn.init.uniform_(self.fr_max_0, 1, 10)
        nn.init.uniform_(self.sigm_slope_0, 0.1, 1.0)
        nn.init.uniform_(self.threshold_0, 0.1, 1.0)
        nn.init.uniform_(self.fr_max_1, 1, 10)
        nn.init.uniform_(self.sigm_slope_1, 0.1, 1.0)
        nn.init.uniform_(self.threshold_1, 0.1, 1.0)
        nn.init.uniform_(self.fr_max_2, 1, 10)
        nn.init.uniform_(self.sigm_slope_2, 0.1, 1.0)
        nn.init.uniform_(self.threshold_2, 0.1, 1.0)
        nn.init.uniform_(self.lfp2lfp_C, -1, 1)
        
        # Firing rate --> LFP connection  -- initialize parameters
        self.fr2lfp_C   = torch.nn.Parameter(torch.Tensor(input_size, input_size))        
        nn.init.uniform_(self.fr2lfp_C, -1, 1)
        # LFP --> Firing rate connection  -- initialize parameters
        self.lfp2fr_C   = torch.nn.Parameter(torch.Tensor(input_size, input_size))        
        nn.init.uniform_(self.lfp2fr_C, -1, 1)
        
    def F(self, x, fr_max, slope, threshold):
        return fr_max * nn.Sigmoid()(slope*(x-threshold)) 
    
    def init_hidden(self, batch_size, delay=1):
        hidden = torch.zeros(batch_size, delay, self.input_size*7, requires_grad=False)
        return hidden       
        
    def getExtInput(self, ExtInput):
        self.ExtInput = ExtInput
    
    def sigm(self, x, fr_max, slope, threshold):
        return fr_max * nn.Sigmoid()(slope*(x-threshold)) 
    
    def forward(self, t, y):
        # Retrieve hidden states for firing rate and LFP
        h_fr, h_lfp1, h_lfp2, h_lfp3, h_lfp4, h_lfp5, h_lfp6 = torch.chunk(y, 7, dim=2)
        # Retrieve external inputs 
        x_fr = self.ExtInput[int(t*1000),:,:,:self.input_size]
        x_lfp = self.ExtInput[int(t*1000),:,:,self.input_size:]
        # Firing rate model
        func_output  = torch.einsum("abc,cd->abd", (x_fr.float(), self.i2o)) + torch.einsum("abc,cd->abd", (h_fr.float(), self.o2o)) + torch.einsum("abc,cd->abd", ((h_lfp2-h_lfp3).float(), self.lfp2fr_C))
        temp_output  = -h_fr + self.F(func_output, self.frmax, self.slope, self.mu)        
        h_fr_next = self.tau*temp_output
        # LFP model (Jasen-Rit model)                
        ext_inp      = torch.einsum("abc,cd->abd", (x_lfp.float(), self.lfp2lfp_C))
        ext_inp_fr   = torch.einsum("abc,cd->abd", (h_fr.float(), self.fr2lfp_C))
        sigm_c1x1 = self.F(self.C1*h_lfp1, self.fr_max_1, self.sigm_slope_1, self.threshold_1)
        sigm_c3x1 = self.F(self.C3*h_lfp1, self.fr_max_2, self.sigm_slope_2, self.threshold_2)
        h_lfp1_next  = h_lfp4 
        h_lfp2_next  = h_lfp5
        h_lfp3_next  = h_lfp6
        h_lfp4_next  = self.A*self.a*(self.F(h_lfp2-h_lfp3, self.fr_max_0, self.sigm_slope_0, self.threshold_0)) - 2*self.a*h_lfp4 - self.a*self.a*h_lfp1
        h_lfp5_next  = self.A*self.a*(self.p_mu+ext_inp+ext_inp_fr+self.C2*sigm_c1x1) - 2*self.a*h_lfp5 - self.a*self.a*h_lfp2
        h_lfp6_next  = self.B*self.b*(self.C4*sigm_c3x1) - 2*self.b*h_lfp6 - self.b*self.b*h_lfp3
        # Combine all the derivative of the hidden states
        dy = torch.cat((h_fr_next, h_lfp1_next, h_lfp2_next, h_lfp3_next, 
                        h_lfp4_next, h_lfp5_next, h_lfp6_next), dim=2)
        return dy

class observation(torch.nn.Module):
    # Observation model used to convert the hidden states value toware output of interest
    def __init__(self, input_size, output_size, bias=False, scale=True):
        super().__init__()
    def forward(self, x):
        h_fr, h_lfp1, h_lfp2, h_lfp3, h_lfp4, h_lfp5, h_lfp6 = torch.chunk(x, 7, dim=-1)
        x_fr = h_fr
        x_lfp = (h_lfp2 - h_lfp3)
        x_new = torch.cat((x_fr, x_lfp), dim=-1)
        return x_new

class LFP_state_estimator(torch.nn.Module):
    # A simple model to estimate the initial state of LFP model
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.lin_W = torch.nn.Linear(input_size, 
                                     output_size*6, bias=bias)
    def forward(self, x):
        return self.lin_W(x)

class RunningAverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
    
#%% Load the data
date = "041423"
makedirs(date)
folder = os.getcwd()
filename = 'airp20220212_04_te1816_trial_dataset.pkl'
with open(folder + '/' + filename, 'rb') as fh:
    dataset = pickle.load(fh)
trial = 0
lfp = dataset['LFP'][0][trial]
spike = dataset['Spikes'][0][trial]

# determine the number of channels utilized for analysis by removing the channels
# less than 10 spikes per seconds
nochannels = []
for trial in dataset['Spikes'][0].keys():
    temp_nochannel = np.where(np.sum(dataset['Spikes'][0][trial], 1)/(dataset['Spikes'][0][trial].shape[1]/1000)<10)[0]
    for tnch in temp_nochannel:
        if tnch not in nochannels:
            nochannels.append(tnch)
desired_channels = [i for i in range(spike.shape[0]) if i not in nochannels]
spike = spike[desired_channels,:]
lfp   = lfp[desired_channels,:]
n_channels = spike.shape[0]
data_size  = spike.shape[1]
# construct the model
func = ODEFunc(n_channels)
emis = observation(n_channels, n_channels)
sest = LFP_state_estimator(n_channels, n_channels)
# set up the hyperparameters
niters     = 1000
log_freq  = 10
batch_size = 1
# start training the model --- for each iteration...
for trial in list(dataset['Spikes'][0].keys()):
    print("Starting trial " + str(trial) + " ...")
    # organize the Firing rate and LFP data 
    lfp = dataset['LFP'][0][trial]
    spike = dataset['Spikes'][0][trial]   
    lfp = lfp[desired_channels,:]
    spike = spike[desired_channels,:]
    firRates = np.zeros(spike.shape)
    for ch in range(len(desired_channels)):
        firRates[ch, :] = spike2fr(spike[ch, :], sigma=0.3)
    
    n_channels = firRates.shape[0]
    data_size  = firRates.shape[1]
    t = torch.linspace(0., data_size-1, data_size)/1000
    # get the ground truths
    true_y = np.concatenate((firRates, lfp), axis=0).T
    true_y = np.reshape(true_y, (true_y.shape[0], 1, true_y.shape[1]))
    true_y = torch.from_numpy(true_y)
    
    # Start the training
    ii = 0
    batch_time = true_y.shape[0] - 1
    optimizer = optim.Adam(list(sest.parameters())+list(func.parameters()), lr=5e-2)
    threshold  = 0.02
    total_train = 0
    
    # keep training until the loss is less than the certain threshold
    while (1):
                   
        total_train += 1
        print("Let's get started for the " + str(total_train) + "th round...")
    
        for itr in range(1, niters + 1):
            optimizer.zero_grad()  # Reset the gradient
            batch_y0, batch_t, batch_y = get_batch(batch_time=batch_time, batch_size=batch_size)  # Get the training data
            spike_batch_y0, lfp_batch_y0 = torch.chunk(batch_y0, 2, dim=-1)
            lfp_batch_y0 = sest(lfp_batch_y0.float())
            batch_y0 = torch.cat((spike_batch_y0.float(), lfp_batch_y0.float()), dim=-1)
            
            func.getExtInput(batch_y.float())
            pred_y = odeint(func, batch_y0, batch_t, method="rk4", options={"step_size": 1e-3})  # Make the predictions
            new_pred_y = emis(pred_y.float())
            
            loss = nn.MSELoss()(batch_y.type(torch.float32), new_pred_y)   # Obtain the loss values   
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            # Calculate the norm loss 
            predictions = new_pred_y.detach().numpy().squeeze()
            groundtruth = batch_y.detach().numpy().squeeze()
            norm_loss = np.mean(np.abs(predictions - groundtruth)/np.max(groundtruth, axis=0), axis=0)

            if itr % log_freq == 0:
                print('Normalize MAE Loss:')
                print(norm_loss)
                fig, axs = plt.subplots(figsize=(15, 8), sharex=True, nrows=5, ncols=2,
                            constrained_layout=True)
                for dim in range(10):
                    i, j = np.unravel_index(dim, (5, 2))
                    axs[i, j].plot(np.linspace(0, batch_time/1e3, batch_time), groundtruth[:, dim],
                                'r--', alpha=0.5)
                    axs[i, j].plot(np.linspace(0, batch_time/1e3, batch_time), predictions[:, dim], 
                              'b--')
                plt.show()
            
                fig, axs = plt.subplots(figsize=(15, 8), sharex=True, nrows=5, ncols=2,
                                        constrained_layout=True)
                for dim in range(10):
                    i, j = np.unravel_index(dim, (5, 2))
                    axs[i, j].plot(np.linspace(0, batch_time/1e3, batch_time), groundtruth[:, 10+dim],
                                'r--', alpha=0.5)
                    axs[i, j].plot(np.linspace(0, batch_time/1e3, batch_time), predictions[:, 10+dim], 
                              'b--')
                plt.show()
            
            loss.backward()
            optimizer.step()  # Update the model parameters
            # Save the current models
            model_file = os.getcwd() + "\\" + date + "\\msDyNODE_sest_trial" + str(trial) + ".pt"
            torch.save(sest.state_dict(), model_file)
            model_file = os.getcwd() + "\\" + date + "\\msDyNODE_trial" + str(trial) + ".pt"
            torch.save(func.state_dict(), model_file)
            model_file = os.getcwd() + "\\" + date + "\\msDyNODE_emis_trial" + str(trial) + ".pt"
            torch.save(emis.state_dict(), model_file)
            # Determine we can stop training the model
            if np.all(norm_loss < threshold):
                break
   