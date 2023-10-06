#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from MF_class import MF_model
from Brian_functions import Brian_run

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["font.size"] = "30"

def read_P_data():
    """
    Read fits of the Transfer Functions (TF) for E/I neurons
    """
    f = open('RS_fit_new_thr_Final.txt', 'r')
    lines = f.readlines()
    PRS=np.zeros(10)
    for i in range(0,len(PRS)):
        PRS[i]=lines[i]
    f.close()


    f = open('FS_fit_new_thr_Final.txt', 'r')
    lines = f.readlines()
    PFS=np.zeros(10)
    for i in range(0,len(PRS)):
        PFS[i]=lines[i]
    f.close()
    
    return PRS, PFS

PRS, PFS = read_P_data()

#Define all parameters

#Network parameters
N1 = 2000; N2 = 8000
prbC = 0.05
Ki = N1*prbC; Ke = N2*prbC

#Izhikevich neurons parameters
gizi = 0.04 ; Eizi = -60
gize = 0.01; Eize = -65
tauize=1; tauizi=1
Tve = 1; Tvi = 1

#adaptation parameters
aFS = 0.; bFS = 0.; cFS = -55; dFS = 0
aRS = 1; bRS = 0.; cRS = -65; dRS = 15
Tue = 1; Tui = 1

#Synaptic current terms
Ee = 0; Ei = -80
Qe = 1.5; Qi = 5.0
Tsyne = 5e-3; Tsyni = 5e-3
tause=5e-3; tausi=5e-3

#MF integration params
feIni=10; fiIni=15

#Input current
Ie = 0; Ii = 0

T = 0.005

def main():
    startTime = datetime.now()
    
    df = 1e-16
    
    #Integrations params
    TotTime = 5.; dt = 0.0001
    percentage = 0.25
    
    
    #####################################
    ###Change connectivity ######
    #####################################
    
    prbC = 0.05
    
    MFparams ={
        "gize": gize, "gizi": gizi, 
        "Eize": Eize, "Eizi": Eizi, 
        "tauize": tauize, "tauizi": tauizi, 
        "Ee": Ee, "Ei": Ei, 
        "tause": tause, "tausi": tausi, 
        "Qe": Qe, "Qi": Qi, 
        "Ke": N2*prbC, "Ki": N1*prbC, 
        "PFS": PFS, "PRS": PRS, 
        "aRS": aRS, "bRS": bRS, "dRS": dRS, 
        "T": T, "N1": N1, "N2": N2
    }

    excitatory_params = [N2, Qe, Tve, Ee, Ei, Ie, gize, Eize, aRS, bRS, cRS, dRS, Tue, Tsyne]
    inhibitory_params = [N1, Qi, Tvi, Ee, Ei, Ii, gizi, Eizi, aFS, bFS, cFS, dFS, Tui, Tsyni]
    
    nu_test = 10#10.0
    nu_cst = np.ones(int(TotTime/dt))*nu_test
    
    feIni = 4; fiIni = 8
    
    #Run MF integration
    MF = MF_model(**MFparams)
    MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_cst, TotTime = TotTime, dt = dt, df = df)
    
    #Run Brian integration
    spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_cst, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
    
    MF_times = MF.t2nd; MF_fe = MF.LSfe2nd; MF_fi = MF.LSfi2nd
    cee = MF.LScee; cii = MF.LScii
    
    Br_times = trajectories[0]; Br_fe = trajectories[3]; Br_fi = trajectories[4]
    
    fig = plt.figure(figsize=(10,8), dpi = 300)
    
    plt.plot(Br_times, Br_fe, '.', c='darkgreen')
    plt.plot(MF_times,MF_fe,c='green',linestyle='--', linewidth=5)
    plt.fill_between(MF_times, MF_fe - np.sqrt(cee), MF_fe + np.sqrt(cee), alpha=0.2, edgecolor='green', facecolor='green')
    
    plt.plot(Br_times, Br_fi, '.', c="darkred")
    plt.plot(MF_times,MF_fi,c='red',linestyle='--', linewidth=5)
    plt.fill_between(MF_times,  MF_fi - np.sqrt(cii),  MF_fi +np.sqrt(cii), alpha=0.2, edgecolor='red', facecolor='red')
    
    
    plt.plot(np.arange(0, TotTime, dt), nu_cst, "y-", lw=3)

    plt.xlabel("time (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.tight_layout()
    plt.show()
    
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()