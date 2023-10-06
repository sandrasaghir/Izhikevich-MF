#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from MF_class import MF_model
from Brian_functions import Brian_run

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["font.size"] = "30"

def heaviside(x):
    return 0.5 * (1 + np.sign(x))

def input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    """
     t1_exc=10. # time of the maximum of external stimulation
     tau1_exc=20. # first time constant of perturbation = rising time
     tau2_exc=50. # decaying time
     ampl_exc=20. # amplitude of excitation
    """
    inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
        heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))+ \
        np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau)))
    return inp

def create_input(t2, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    """
    Create a time dependent input for the network
    """
    test_input = []
    for ji in t2:
        test_input.append(6.+input_rate(ji, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau))
    return test_input

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
nueIni=10; nuiIni=15

#Input current
Ie = 0; Ii = 0

T = 0.005

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

def main():
    startTime = datetime.now()
    
    df = 1e-16
    
    #Integrations params
    TotTime = 5.; dt = 0.0001
    percentage = 0.25
    
    t = np.arange(0, TotTime*10**3, dt*10**3)
    
    #############################
    ##Constant input############
    #############################
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
    
    str2save = './plots/fig4/nu_cst/'
    
    np.savetxt(str2save + 'drive', nu_cst)
    
    np.savetxt(str2save + 'MF_times', MF_times)
    np.savetxt(str2save + 'MF_fe', MF_fe)
    np.savetxt(str2save + 'MF_fi', MF_fi)
    np.savetxt(str2save + 'MF_cee', cee)
    np.savetxt(str2save + 'MF_cii', cii)
    
    np.savetxt(str2save + 'Br_times', Br_times)
    np.savetxt(str2save + 'Br_fe', Br_fe)
    np.savetxt(str2save + 'Br_fi', Br_fi)
    
    
    fig = plt.figure(figsize=(10,8), dpi = 300)

    plt.plot(Br_times, Br_fe, '-', c='green', alpha=0.5)
    plt.plot(MF_times,MF_fe,c='darkgreen',linestyle='--', linewidth=5)
    plt.fill_between(MF_times, MF_fe - np.sqrt(cee), MF_fe + np.sqrt(cee), alpha=0.2, edgecolor='green', facecolor='green')

    plt.plot(Br_times, Br_fi, '-', c="red", alpha=0.5)
    plt.plot(MF_times,MF_fi,c='darkred',linestyle='--', linewidth=5)
    plt.fill_between(MF_times,  MF_fi - np.sqrt(cii),  MF_fi +np.sqrt(cii), alpha=0.2, edgecolor='red', facecolor='red')

    plt.plot(MF_times, nu_cst, "y-", lw=3)

    plt.xlabel("time (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.tight_layout()
    plt.show()
    
    #############################
    ##Sharp input############
    #############################
    t1_exc=2500. # time of the maximum of external stimulation
    tau1_exc= 20. # first time constant of perturbation = rising time
    tau2_exc= 20. # decaying time
    ampl_exc= 20. # amplitude of excitation
    plateau = 20
    #minimal drive needed or MF breaks
    nu_cst = np.ones(int(TotTime/dt))*6
    nu_sharp = input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau)
    nu_sharp += nu_cst
    
    feIni = 4; fiIni = 8
    
    
    #Run MF integration
    MF = MF_model(**MFparams)
    MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_sharp, TotTime = TotTime, dt = dt, df = df)
    
    #Run Brian integration
    spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_sharp, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
    
    MF_times = MF.t2nd; MF_fe = MF.LSfe2nd; MF_fi = MF.LSfi2nd
    cee = MF.LScee; cii = MF.LScii
    
    Br_times = trajectories[0]; Br_fe = trajectories[3]; Br_fi = trajectories[4]
    
    str2save = './plots/fig4/nu_sharp/'
    
    np.savetxt(str2save + 'drive', nu_sharp)
    
    np.savetxt(str2save + 'MF_times', MF_times)
    np.savetxt(str2save + 'MF_fe', MF_fe)
    np.savetxt(str2save + 'MF_fi', MF_fi)
    np.savetxt(str2save + 'MF_cee', cee)
    np.savetxt(str2save + 'MF_cii', cii)
    
    np.savetxt(str2save + 'Br_times', Br_times)
    np.savetxt(str2save + 'Br_fe', Br_fe)
    np.savetxt(str2save + 'Br_fi', Br_fi)
    
    
    fig = plt.figure(figsize=(10,8), dpi = 300)

    plt.plot(Br_times, Br_fe, '-', c='green', alpha=0.5)
    plt.plot(MF_times,MF_fe,c='darkgreen',linestyle='--', linewidth=5)
    plt.fill_between(MF_times, MF_fe - np.sqrt(cee), MF_fe + np.sqrt(cee), alpha=0.2, edgecolor='green', facecolor='green')

    plt.plot(Br_times, Br_fi, '-', c="red", alpha=0.5)
    plt.plot(MF_times,MF_fi,c='darkred',linestyle='--', linewidth=5)
    plt.fill_between(MF_times,  MF_fi - np.sqrt(cii),  MF_fi +np.sqrt(cii), alpha=0.2, edgecolor='red', facecolor='red')

    plt.plot(MF_times, nu_sharp, "y-", lw=3)

    plt.xlabel("time (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.tight_layout()
    plt.show()
    
    #############################
    ##Wide input############
    #############################
    t1_exc=2500. # time of the maximum of external stimulation
    tau1_exc=700. # first time constant of perturbation = rising time
    tau2_exc=700. # decaying time
    ampl_exc= 10. # amplitude of excitation
    plateau = 20
    
    #minimal drive needed or MF breaks
    nu_cst = np.ones(int(TotTime/dt))*6
    nu_wide = input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau)
    nu_wide += nu_cst
    
    feIni = 4; fiIni = 8
    
    #Run MF integration
    MF = MF_model(**MFparams)
    MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_wide, TotTime = TotTime, dt = dt, df = df)
    
    #Run Brian integration
    spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_wide, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
    
    MF_times = MF.t2nd; MF_fe = MF.LSfe2nd; MF_fi = MF.LSfi2nd
    cee = MF.LScee; cii = MF.LScii
    
    Br_times = trajectories[0]; Br_fe = trajectories[3]; Br_fi = trajectories[4]
    
    str2save = './plots/fig4/nu_wide/'
    
    np.savetxt(str2save + 'drive', nu_wide)
    
    np.savetxt(str2save + 'MF_times', MF_times)
    np.savetxt(str2save + 'MF_fe', MF_fe)
    np.savetxt(str2save + 'MF_fi', MF_fi)
    np.savetxt(str2save + 'MF_cee', cee)
    np.savetxt(str2save + 'MF_cii', cii)
    
    np.savetxt(str2save + 'Br_times', Br_times)
    np.savetxt(str2save + 'Br_fe', Br_fe)
    np.savetxt(str2save + 'Br_fi', Br_fi)
    
    
    fig = plt.figure(figsize=(10,8), dpi = 300)

    plt.plot(Br_times, Br_fe, '-', c='green', alpha=0.5)
    plt.plot(MF_times,MF_fe,c='darkgreen',linestyle='--', linewidth=5)
    plt.fill_between(MF_times, MF_fe - np.sqrt(cee), MF_fe + np.sqrt(cee), alpha=0.2, edgecolor='green', facecolor='green')

    plt.plot(Br_times, Br_fi, '-', c="red", alpha=0.5)
    plt.plot(MF_times,MF_fi,c='darkred',linestyle='--', linewidth=5)
    plt.fill_between(MF_times,  MF_fi - np.sqrt(cii),  MF_fi +np.sqrt(cii), alpha=0.2, edgecolor='red', facecolor='red')

    plt.plot(MF_times, nu_wide, "y-", lw=3)

    plt.xlabel("time (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.tight_layout()
    plt.show()
    
    #############################
    ##Slowly decaying input############
    #############################
    t1_exc=50. # time of the maximum of external stimulation
    tau1_exc= 10. # first time constant of perturbation = rising time
    tau2_exc= 1500. # decaying time
    ampl_exc= 25. # amplitude of excitation
    plateau = 30
    
    nu_cst = np.ones(int(TotTime/dt))*6
    nu_decay = input_rate(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau)
    nu_decay += nu_cst
    
    feIni = 4; fiIni = 8
    
    #Run MF integration
    MF = MF_model(**MFparams)
    MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_decay, TotTime = TotTime, dt = dt, df = df)
    
    #Run Brian integration
    spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_decay, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
    
    MF_times = MF.t2nd; MF_fe = MF.LSfe2nd; MF_fi = MF.LSfi2nd
    cee = MF.LScee; cii = MF.LScii
    
    Br_times = trajectories[0]; Br_fe = trajectories[3]; Br_fi = trajectories[4]
    
    str2save = './plots/fig4/nu_decay/'
    
    np.savetxt(str2save + 'drive', nu_decay)
    
    np.savetxt(str2save + 'MF_times', MF_times)
    np.savetxt(str2save + 'MF_fe', MF_fe)
    np.savetxt(str2save + 'MF_fi', MF_fi)
    np.savetxt(str2save + 'MF_cee', cee)
    np.savetxt(str2save + 'MF_cii', cii)
    
    np.savetxt(str2save + 'Br_times', Br_times)
    np.savetxt(str2save + 'Br_fe', Br_fe)
    np.savetxt(str2save + 'Br_fi', Br_fi)
    
    
    fig = plt.figure(figsize=(10,8), dpi = 300)

    plt.plot(Br_times, Br_fe, '-', c='green', alpha=0.5)
    plt.plot(MF_times,MF_fe,c='darkgreen',linestyle='--', linewidth=5)
    plt.fill_between(MF_times, MF_fe - np.sqrt(cee), MF_fe + np.sqrt(cee), alpha=0.2, edgecolor='green', facecolor='green')

    plt.plot(Br_times, Br_fi, '-', c="red", alpha=0.5)
    plt.plot(MF_times,MF_fi,c='darkred',linestyle='--', linewidth=5)
    plt.fill_between(MF_times,  MF_fi - np.sqrt(cii),  MF_fi +np.sqrt(cii), alpha=0.2, edgecolor='red', facecolor='red')

    plt.plot(MF_times, nu_decay, "y-", lw=3)

    plt.xlabel("time (s)")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.tight_layout()
    plt.show()

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
    
if __name__ == '__main__':
    main()