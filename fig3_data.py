#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from MF_class import MF_model
from tqdm import tqdm
from Brian_functions import Brian_run
import scipy.stats
import matplotlib

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

def main():
    PRS, PFS = read_P_data()

    startTime = datetime.now()
    #Integrations params
    TotTime = 5.; dt = 0.0001
    percentage = 0.25
    nu_test = 10#10.0
    nu_ext = np.ones(int(TotTime/dt))*nu_test
    df = 1e-16
    feIni = 4; fiIni = 8
    
    ##################################################################
    print("Checking external drive effects...")
    dRS = 15
    
    nu_list = np.linspace(1, 20, 20)
    
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/nu_values.txt", nu_list)
    
    #Storage
    Br_mean_nue_list = np.zeros(len(nu_list)); Br_mean_nui_list = np.zeros(len(nu_list))
    Br_std_dev_nue_list = np.zeros(len(nu_list)); Br_std_dev_nui_list = np.zeros(len(nu_list))
    
    MF_mean_nue_list = np.zeros(len(nu_list)); MF_mean_nui_list = np.zeros(len(nu_list))
    cee_list = np.zeros(len(nu_list)); cii_list = np.zeros(len(nu_list))
    
    
    for i in tqdm(range(len(nu_list))):
        #Update integration parameters
        nu_test = nu_list[i]
        nu_ext = np.ones(int(TotTime/dt))*nu_test
        #Update MF
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
        
        #Update brian
        excitatory_params = [N2, Qe, Tve, Ee, Ei, Ie, gize, Eize, aRS, bRS, cRS, dRS, Tue, Tsyne]
        inhibitory_params = [N1, Qi, Tvi, Ee, Ei, Ii, gizi, Eizi, aFS, bFS, cFS, dFS, Tui, Tsyni]
    
        #Run MF integration
        MF = MF_model(**MFparams)
        MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_ext, TotTime = TotTime, dt = dt, df = df)
        
        #Run Brian integration
        spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_ext, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
        
        #Get MF data
        MF_mean_nue_list[i] = MF.LSfe2nd[-1]; MF_mean_nui_list[i] = MF.LSfi2nd[-1]
        cee_list[i] = np.sqrt(MF.LScee[-1]); cii_list[i] = np.sqrt(MF.LScii[-1])
        
        #Get Brian data
        Br_mean_nue_list[i] = mean_results[2]; Br_mean_nui_list[i] = mean_results[3]
        Br_std_dev_nue_list[i] = std_devs[2]; Br_std_dev_nui_list[i] = std_devs[3]
        
        #Save two sample distributions (histogram)
        if nu_test == 10 or nu_test == 15:
            
            BrpopRateG_exc = trajectories[3]
            BrpopRateG_inh = trajectories[4]
            
            BrpopRateG_exc = BrpopRateG_exc[int(percentage*len(BrpopRateG_exc))::]
            BrpopRateG_inh = BrpopRateG_inh[int(percentage*len(BrpopRateG_inh))::]
            
            np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/distribution_nu=" + str(nu_test) + "/Br_poprate_inh.txt", BrpopRateG_inh)
            np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/distribution_nu=" + str(nu_test) + "/Br_poprate_exc.txt",BrpopRateG_exc)
            
            mean_FRE = MF.LSfe2nd[-1]; mean_FRI = MF.LSfi2nd[-1]
            std_FRE = np.sqrt(MF.LScee[-1]); std_FRI = np.sqrt(MF.LScii[-1])
            
            np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/distribution_nu=" + str(nu_test) + "/MF_means_stdvs.txt", [mean_FRE, mean_FRI, std_FRE, std_FRI])
            
            #Plot
            nu_range = np.linspace(0, 20, 1000)
            MFGaussE = scipy.stats.norm.pdf(nu_range, mean_FRE, std_FRE)
            MFGaussI = scipy.stats.norm.pdf(nu_range, mean_FRI, std_FRI)
            
            plt.figure(dpi=600)
            plt.hist(BrpopRateG_inh, bins='auto', density=True, color="red", edgecolor = "black")
            plt.hist(BrpopRateG_exc, bins='auto', density=True, color="green", edgecolor = "black")
            plt.plot(nu_range, MFGaussI, "r-", lw = 3)
            plt.plot(nu_range, MFGaussE, "g-", lw = 3)
            plt.fill_between(nu_range, MFGaussI, color="red", alpha=0.2)
            plt.fill_between(nu_range, MFGaussE, color="green", alpha=0.2)
            plt.title("Repartition of the firing rates - constant input: nu =" + str(nu_test))
            plt.xlabel(r"$\nu$")
            
            plt.savefig("./results/external_drive_effects/dRS=" + str(dRS) + "/distribution_nu=" + str(nu_test) + "/distribution.png")
            plt.show()
            
            
    
    #save MF data
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/MF_fe.txt", MF_mean_nue_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/MF_fi.txt", MF_mean_nui_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/MF_cee.txt", cee_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/MF_cii.txt", cii_list)
    
    #Save Brian data
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/Br_fe.txt", Br_mean_nue_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) +"/Br_fi.txt", Br_mean_nui_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/Br_stdv_fe.txt", Br_std_dev_nue_list)
    np.savetxt("./results/external_drive_effects/dRS=" + str(dRS) + "/Br_stdv_fi.txt", Br_std_dev_nui_list)
    
    #Plot to check
    
    plt.figure(figsize=(8, 6), dpi=80)
    
    #Brian fe
    plt.plot(nu_list, Br_mean_nue_list, "g-", label = "Brian - Excitatory")
    plt.plot(nu_list, Br_mean_nui_list, "r-", label = "Brian - Inhibitory")
    
    #MF fe
    plt.plot(nu_list, MF_mean_nue_list, "g--", label = "MF - Excitatory")
    plt.plot(nu_list, MF_mean_nui_list, "r--", label = "MF - Inhibitory")
    
    #Brian stdv
    plt.errorbar(nu_list, Br_mean_nue_list, yerr = Br_std_dev_nue_list, fmt = '+',color = 'green')
    plt.errorbar(nu_list, Br_mean_nui_list, yerr = Br_std_dev_nui_list, fmt = '+',color = 'red')
    
    plt.plot(nu_list, MF_mean_nue_list, "go")
    plt.plot(nu_list, MF_mean_nui_list, "ro")
    
    plt.fill_between(nu_list, MF_mean_nue_list - np.sqrt(cee_list), MF_mean_nue_list + np.sqrt(cee_list),
                 color='green', alpha=0.2)
    plt.fill_between(nu_list, MF_mean_nui_list - np.sqrt(cii_list), MF_mean_nui_list + np.sqrt(cii_list),
                 color='red', alpha=0.2)
    
    plt.xlabel(r"$\nu_{ext}$")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.title("External drive effect on the mean firing rates of the populations - constant input frequency")
    plt.legend()
    #plt.savefig("./results/adaptation_effects/nuext=" + str(nu_test) + "/nu_vs_d")
    plt.show()
    
    
    ##################################################################
    print("Checking adaptation effects...")
    #Choose a drive
    nu_test = 10
    nu_ext = np.ones(int(TotTime/dt))*nu_test
    #Adaptation parameter effect
    d_list = np.linspace(0, 30, 31)

    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/d_values.txt", d_list)
    
    #Storage
    Br_mean_nue_list = np.zeros(len(d_list)); Br_mean_nui_list = np.zeros(len(d_list))
    Br_std_dev_nue_list = np.zeros(len(d_list)); Br_std_dev_nui_list = np.zeros(len(d_list))
    
    MF_mean_nue_list = np.zeros(len(d_list)); MF_mean_nui_list = np.zeros(len(d_list))
    cee_list = np.zeros(len(d_list)); cii_list = np.zeros(len(d_list))
    for i in tqdm(range(len(d_list))):
        #Update integration parameters
        dRS = d_list[i]
        #Update MF
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
        
        #Update brian
        excitatory_params = [N2, Qe, Tve, Ee, Ei, Ie, gize, Eize, aRS, bRS, cRS, dRS, Tue, Tsyne]
        inhibitory_params = [N1, Qi, Tvi, Ee, Ei, Ii, gizi, Eizi, aFS, bFS, cFS, dFS, Tui, Tsyni]
    
        #Run MF integration
        MF = MF_model(**MFparams)
        MF.second_order(feIni = feIni, fiIni = fiIni, external_input = nu_ext, TotTime = TotTime, dt = dt, df = df)
        
        #Run Brian integration
        spikes, trajectories, mean_results, std_devs = Brian_run(prbC, nu_ext, excitatory_params, inhibitory_params, TotTime, dt, T, percentage)
        
        #Get MF data
        MF_mean_nue_list[i] = MF.LSfe2nd[-1]; MF_mean_nui_list[i] = MF.LSfi2nd[-1]
        cee_list[i] = np.sqrt(MF.LScee[-1]); cii_list[i] = np.sqrt(MF.LScii[-1])
        
        #Get Brian data
        Br_mean_nue_list[i] = mean_results[2]; Br_mean_nui_list[i] = mean_results[3]
        Br_std_dev_nue_list[i] = std_devs[2]; Br_std_dev_nui_list[i] = std_devs[3]
        
        #Save two sample distributions (histogram)
        if dRS == 10 or dRS == 20:
            
            BrpopRateG_exc = trajectories[3]
            BrpopRateG_inh = trajectories[4]
            
            BrpopRateG_exc = BrpopRateG_exc[int(percentage*len(BrpopRateG_exc))::]
            BrpopRateG_inh = BrpopRateG_inh[int(percentage*len(BrpopRateG_inh))::]
            
            np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/distribution_d=" + str(dRS) + "/Br_poprate_inh.txt", BrpopRateG_inh)
            np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/distribution_d=" + str(dRS) + "/Br_poprate_exc.txt",BrpopRateG_exc)
            
            mean_FRE = MF.LSfe2nd[-1]; mean_FRI = MF.LSfi2nd[-1]
            std_FRE = np.sqrt(MF.LScee[-1]); std_FRI = np.sqrt(MF.LScii[-1])
            
            np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/distribution_d=" + str(dRS) + "/MF_means_stdvs.txt", [mean_FRE, mean_FRI, std_FRE, std_FRI])
            
            #Plot
            nu_range = np.linspace(0, 20, 1000)
            MFGaussE = scipy.stats.norm.pdf(nu_range, mean_FRE, std_FRE)
            MFGaussI = scipy.stats.norm.pdf(nu_range, mean_FRI, std_FRI)
            
            plt.figure(dpi=600)
            plt.hist(BrpopRateG_inh, bins='auto', density=True, color="red", edgecolor = "black")
            plt.hist(BrpopRateG_exc, bins='auto', density=True, color="green", edgecolor = "black")
            plt.plot(nu_range, MFGaussI, "r-", lw = 3)
            plt.plot(nu_range, MFGaussE, "g-", lw = 3)
            plt.fill_between(nu_range, MFGaussI, color="red", alpha=0.2)
            plt.fill_between(nu_range, MFGaussE, color="green", alpha=0.2)
            plt.title("Repartition of the firing rates - constant input")
            plt.xlabel(r"$\nu$")
            
            #plt.savefig("./results/adaptation_effects/nuext=" + str(nu_test) + "/distribution_d=" + str(dRS) + "/distribution.png")
            plt.show()
            
            
    
    #save MF data
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/MF_fe.txt", MF_mean_nue_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/MF_fi.txt", MF_mean_nui_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/MF_cee.txt", cee_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/MF_cii.txt", cii_list)
    
    #Save Brian data
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/Br_fe.txt", Br_mean_nue_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/Br_fi.txt", Br_mean_nui_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/Br_stdv_fe.txt", Br_std_dev_nue_list)
    np.savetxt("./results/adaptation_effects/nuext=" + str(nu_test) + "/Br_stdv_fi.txt", Br_std_dev_nui_list)
    
    #Plot to check
    
    plt.figure(figsize=(8, 6), dpi=80)
    
    #Brian fe
    plt.plot(d_list, Br_mean_nue_list, "g-", label = "Brian - Excitatory")
    plt.plot(d_list, Br_mean_nui_list, "r-", label = "Brian - Inhibitory")
    
    #MF fe
    plt.plot(d_list, MF_mean_nue_list, "g--", label = "MF - Excitatory")
    plt.plot(d_list, MF_mean_nui_list, "r--", label = "MF - Inhibitory")
    
    #Brian stdv
    plt.errorbar(d_list, Br_mean_nue_list, yerr = Br_std_dev_nue_list, fmt = '+',color = 'green')
    plt.errorbar(d_list, Br_mean_nui_list, yerr = Br_std_dev_nui_list, fmt = '+',color = 'red')
    
    plt.plot(d_list, MF_mean_nue_list, "go")
    plt.plot(d_list, MF_mean_nui_list, "ro")
    
    plt.fill_between(d_list, MF_mean_nue_list - np.sqrt(cee_list), MF_mean_nue_list + np.sqrt(cee_list),
                 color='green', alpha=0.2)
    plt.fill_between(d_list, MF_mean_nui_list - np.sqrt(cii_list), MF_mean_nui_list + np.sqrt(cii_list),
                 color='red', alpha=0.2)
    
    plt.xlabel("d value")
    plt.ylabel(r"$\nu$ (Hz)")
    plt.title("Adaptation parameter effect on the mean firing rates of the populations - constant input frequency")
    plt.legend()
    #plt.savefig("./results/adaptation_effects/nuext=" + str(nu_test) + "/nu_vs_d.png")
    plt.show()

    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

if __name__ == '__main__':
    main()
