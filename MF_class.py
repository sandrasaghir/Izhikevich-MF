#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp_spec
from tqdm import tqdm

class MF_model():
    
    def __init__(self, gize, gizi, Eize, Eizi, tauize, tauizi, Ee, Ei, tause, tausi,\
        Qe, Qi, Ke, Ki, PFS, PRS, aRS, bRS, dRS, T, N1, N2):
        
        #Izhikevich neuron parameters
        self.gize = gize; self.gizi = gizi
        self.Eize = Eize; self.Eizi = Eizi
        self.tauize = tauize; self.tauizi = tauizi
        
        #Synaptic current parameters
        #Equilibrium potential
        self.Ee = Ee; self.Ei = Ei
        #Synaptic time constants
        self.tause = tause; self.tausi = tausi
        
        #adaptation parameters of the E (RS) neurons 
        self.aRS = aRS; self.bRS = bRS; self.dRS = dRS
        
        #quantal increments of the conductance
        self.Qi = Qi; self.Qe = Qe
        
        #Network connectivity
        self.Ke = Ke; self.Ki = Ki
        
        #Fits for the transfer function
        self.PRS = PRS
        self.PFS = PFS
        
        #Time constant of the firing rates: assuming Markovian dynamics
        self.T = T
        
        #Number of neurons in the initial populations
        self.N1 = N1; self.N2 = N2
        
    def update_TFs(self, fe, fi, ue, ui, compute_stdg = False):
        """
        Update the transfer function value with current firing rates;
        The right order is:
            -> average membrane conductances
            -> std of membrane conductances
            -> average membrane voltages
            -> membrane voltages standard deviations
            -> membrane voltages autocorrelation times
            -> transfer functions
        """
        
        #Average conductances
        self.muge = self.Qe*self.tause*self.Ke*fe
        self.mugi = self.Qi*self.tausi*self.Ki*fi
        
        if compute_stdg == True:
            if fi < 0:
                print("NOT OK")
            #Conductances stdv
            self.stdge = np.sqrt(self.tause*self.Ke*fe/2)*self.Qe
            self.stdgi = np.sqrt(self.tausi*self.Ki*fi/2)*self.Qi
        
        #Average voltages
        self.muv(ue, ui)
        
        #Voltages stdv
        self.stdv(fe, fi)
        
        #Voltages autocorrelation times
        self.tauv(fe, fi)
        
        #TFs
        return self.TransferFunction()
    
    def muv(self, ue, ui):
        """
        Average membrane voltage of populations E and I
        """
            
        self.muve = ((2*self.gize*self.Eize + self.muge + self.mugi + self.bRS) - \
        np.sqrt((2*self.gize*self.Eize + self.muge + self.mugi + self.bRS)**2 -\
        4*self.gize*(self.gize*self.Eize**2 + self.muge*self.Ee + self.mugi*self.Ei - ue)))/(2*self.gize)

        self.muvi =((2*self.gizi*self.Eizi+self.muge+self.mugi)\
        -np.sqrt((2*self.gizi*self.Eizi+self.muge+self.mugi)**2\
        -4*self.gizi*(self.gizi*self.Eizi**2+self.muge*self.Ee+self.mugi*self.Ei-ui)))/(2*self.gizi)
    
    def stdv(self, fe, fi):
        """
        Stdv of membrane voltage of populations E and I
        """
        
        #Excitatory pop
        ae = self.gize*(self.muve - self.Eize)**2
        ai = self.gize*(self.muve - self.Eize)**2
     
        be = self.Qe*(self.Ee - self.muve)
        bi = self.Qi*(self.Ei - self.muve)

        ce = self.tause
        ci = self.tausi
        
        argsve = self.Ke*fe*(2*ae*be*ce**3/self.tauize**2 + ce**3*be**2/(8*self.tauize) \
        + be**2*ce**3/(8*self.tauize**2)) + self.Ki*fi*(2*ai*bi*ci**3/self.tauizi**2 \
        + ci**3*bi**2/(8*self.tauizi) + bi**2*ci**3/(8*self.tauizi**2))

        if argsve > 0:
            self.stdve = np.sqrt(argsve)
        else:
            self.stdve = 1e-9
        
        #Inhibitory pop
        ae = self.gizi*(self.muvi - self.Eizi)**2
        ai = self.gizi*(self.muvi - self.Eizi)**2
     
        be = self.Qe*(self.Ee - self.muvi)
        bi = self.Qi*(self.Ei - self.muvi)

        ce = self.tause
        ci = self.tausi
        
        argsvi = self.Ke*fe*(2*ae*be*ce**3/self.tauize**2 + ce**3*be**2/(8*self.tauize) \
        + be**2*ce**3/(8*self.tauize**2)) + self.Ki*fi*(2*ai*bi*ci**3/self.tauizi**2 \
        + ci**3*bi**2/(8*self.tauizi) + bi**2*ci**3/(8*self.tauizi**2))

        if argsvi > 0:
            self.stdvi = np.sqrt(argsvi)
        else:
            self.stdvi = 1e-9

    def tauv(self, fe, fi):
        """
        Membrane auto-correlation time
        """
        
        fe = fe + 1e-9
        fi = fi + 1e-9

        #Excitatory pop
        ae = self.gize*(self.muve - self.Eize)**2
        ai = self.gize*(self.muve - self.Eize)**2
     
        be = self.Qe*(self.Ee - self.muve)
        bi = self.Qi*(self.Ei - self.muve)

        ce = self.tause
        ci = self.tausi
      
        argve = self.Ke*fe*(2*ae*be*ce**3/self.tauize**2 + ce**3*be**2/(8*self.tauize)\
        + be**2*ce**3/(8*self.tauize**2)) + self.Ki*fi*(2*ai*bi*ci**3/self.tauizi**2\
        + ci**3*bi**2/(8*self.tauizi) + bi**2*ci**3/(8*self.tauizi**2))
        
        if argve > 0:
            numve = 0.5*(self.Ke*fe*(be**2*ce**4/(2*np.pi*self.tauize**2))\
            + self.Ki*fi*(bi**2*ci**4/(2*np.pi*self.tauizi**2)))
            
            self.tauve = numve/(argve + 1e-9)
            
        else:
            self.tauve = 1
          
        #Inhibitory pop
        ae = self.gizi*(self.muvi - self.Eizi)**2
        ai = self.gizi*(self.muvi - self.Eizi)**2
     
        be = self.Qe*(self.Ee - self.muvi)
        bi = self.Qi*(self.Ei - self.muvi)

        ce = self.tause
        ci = self.tausi
        
        argvi = self.Ke*fe*(2*ae*be*ce**3/self.tauize**2 + ce**3*be**2/(8*self.tauize)\
        + be**2*ce**3/(8*self.tauize**2)) + self.Ki*fi*(2*ai*bi*ci**3/self.tauizi**2\
        + ci**3*bi**2/(8*self.tauizi) + bi**2*ci**3/(8*self.tauizi**2))
        
        if argvi > 0:
            numvi = 0.5*(self.Ke*fe*(be**2*ce**4/(2*np.pi*self.tauize**2))\
            + self.Ki*fi*(bi**2*ci**4/(2*np.pi*self.tauizi**2)))
            
            self.tauvi = numvi/(argvi + 1e-9)
            
        else:
            self.tauvi = 1
    
    def TransferFunction(self):
        """
        Transfer function of the neurons
        """
        #TF fit parameters
        muvo=-30#5
        dmuvo=20#25
        svo=0.7
        dsvo=1
        tauvo=0.0015
        dtauvo=0.003
        
        Pscale=1.
        
        #Excitatory pop
        Po, Pmuv, Psv, Ptauv, Pvsv, Pvtauv, Psvtauv, Pvv, Ptt, Pss = self.PRS
        
        TFe = Pscale*(sp_spec.erfc(((Po + Pmuv*(self.muve - muvo)/dmuvo + Psv*(self.stdve - svo)/dsvo\
        + Ptauv*(self.tauve - tauvo)/dtauvo + Pvsv*(self.muve - muvo)*(self.stdve - svo)/(dsvo*dmuvo)\
        + Pvtauv*(self.muve - muvo)*(self.tauve - tauvo)/(dtauvo*dmuvo) + Psvtauv*(self.stdve - svo)*(self.tauve - tauvo)/(dtauvo*dsvo)\
        + Pvv*(self.muve - muvo)**2/(dmuvo*dmuvo) + Ptt*(self.tauve - tauvo)**2/(dtauvo*dtauvo)\
        + Pss*(self.stdve - svo)**2/(dsvo*dsvo)) - self.muve)/(np.sqrt(2)*self.stdve) ))/(2*self.tauve)
     
        #Inhibitory pop
        Po, Pmuv, Psv, Ptauv, Pvsv, Pvtauv, Psvtauv, Pvv, Ptt, Pss = self.PFS
        
        TFi = Pscale*(sp_spec.erfc(((Po + Pmuv*(self.muvi - muvo)/dmuvo + Psv*(self.stdvi - svo)/dsvo\
        + Ptauv*(self.tauvi - tauvo)/dtauvo + Pvsv*(self.muvi - muvo)*(self.stdvi - svo)/(dsvo*dmuvo)\
        + Pvtauv*(self.muvi - muvo)*(self.tauvi - tauvo)/(dtauvo*dmuvo) + Psvtauv*(self.stdvi - svo)*(self.tauvi - tauvo)/(dtauvo*dsvo)\
        + Pvv*(self.muvi - muvo)**2/(dmuvo*dmuvo) + Ptt*(self.tauvi - tauvo)**2/(dtauvo*dtauvo)\
        + Pss*(self.stdvi - svo)**2/(dsvo*dsvo)) - self.muvi)/(np.sqrt(2)*self.stdvi) ))/(2*self.tauvi)
            
        return TFe, TFi
    
    def first_order(self, feIni, fiIni, external_input, TotTime, dt):
        """
        First order integration of the MF equations
        
        feIni, fiIni: initial firing rates
        external_input: firing rate of the external drive. int, float or array
        TotTime: integration duration 
        dt: timestep
        
        
        Returns arrays containing the time evolutions of
        -> Ue the recovery variable
        -> fe, fi the firing rates
        -> muge, mugi the average conductances
        -> stdge, stdgi the std of the conductances
        -> muve, muvi the average voltages
        -> stdve, stdvi the std of the voltages
        """
        
        #Duration
        nSteps =  int(TotTime/dt)
        self.t = np.linspace(0, TotTime, nSteps)
        
        #Turn external input into array if constant
        if type(external_input) == float or type(external_input) == int:
            external_input = np.ones(nSteps)*external_input
        
        #Initialize
        fecont = feIni; ficont = fiIni
        Ue = fecont*(self.dRS/self.aRS)
        Ui = 0
        
        #Prepare lists
        self.LSUe = np.empty(nSteps)
        
        #firing rates
        self.LSfe = np.empty(nSteps)
        self.LSfi = np.empty(nSteps)
        
        #membrane conductances
        self.LSmuge = np.empty(nSteps)
        self.LSmugi = np.empty(nSteps)
        self.LSstdge = np.empty(nSteps)
        self.LSstdgi = np.empty(nSteps)
        
        #membrane potentials
        self.LSmuve = np.empty(nSteps)
        self.LSmuvi = np.empty(nSteps)
        self.LSstdve = np.empty(nSteps)
        self.LSstdvi = np.empty(nSteps)
        
        #Main loop
        for i in range(len(self.t)):
            
            #Update the population parameters with the current firing rates
            TFe, TFi = self.update_TFs(fecont + external_input[i], ficont, Ue, Ui, compute_stdg = True)
            
            #update u with current fe value
            Ue += dt*(-self.aRS*Ue + (self.dRS)*fecont)
            
            #Update fe and fi
            fecont += (dt/self.T)*(TFe - fecont) 
            ficont += (dt/self.T)*(TFi - ficont) 
            
            #Store
            #Firing rates
            self.LSfe[i] = fecont; self.LSfi[i] = ficont
            #Recovery
            self.LSUe[i] = Ue
            #Conductances
            self.LSmuge[i] = self.muge
            self.LSmugi[i] = self.mugi
            self.LSstdge[i] = self.stdge
            self.LSstdgi[i] = self.stdgi
            #Voltages
            self.LSmuve[i] = self.muve
            self.LSmuvi[i] = self.muvi
            self.LSstdve[i] = self.stdve
            self.LSstdvi[i] = self.stdvi

    def diffTF(self, fe, fi, Ue, Ui, df):
        """
        Numerical differentiations of TF around the current state of the system
        
        fe, fi, Ue, Ui: state coordinates
        df: step size
        """
        #First derivatives
        
        #Infinitesimal steps in the fe direction
        TFe_fe_plus, TFi_fe_plus = self.update_TFs(fe + df, fi, Ue, Ui)
        TFe_fe_minus, TFi_fe_minus = self.update_TFs(fe - df, fi, Ue, Ui)
        
        #Infinitesimal steps in the fi direction
        TFe_fi_plus, TFi_fi_plus = self.update_TFs(fe, fi + df, Ue, Ui)
        TFe_fi_minus, TFi_fi_minus = self.update_TFs(fe, fi - df, Ue, Ui)
        
        #dTFe/dfe
        self._diff_TFe_fe = (TFe_fe_plus - TFe_fe_minus)/(2*df)
        #dTFi/dfe
        self._diff_TFi_fe = (TFi_fe_plus - TFi_fe_minus)/(2*df)
        #dTFe/dfi
        self._diff_TFe_fi = (TFe_fi_plus - TFe_fi_minus)/(2*df)
        #dTFi/dfi
        self._diff_TFi_fi = (TFi_fi_plus - TFi_fi_minus)/(2*df)
        
        #Second derivatives
        
        #d2TFe/dfe2
        self._diff2_TFe_fe_fe = (TFe_fe_plus - 2*self.TFe + TFe_fe_minus)/((df)**2)
        #d2TFi/dfe2
        self._diff2_TFi_fe_fe = (TFi_fe_plus - 2*self.TFi + TFi_fe_minus)/((df)**2)
        #d2TFe/dfi2
        self._diff2_TFe_fi_fi = (TFe_fi_plus - 2*self.TFe + TFe_fi_minus)/((df)**2)
        #d2TFi/dfi2
        self._diff2_TFi_fi_fi = (TFi_fi_plus - 2*self.TFi + TFi_fi_minus)/((df)**2)
        
        #Cross derivatives
        TFe_fe_plus_fi_plus, TFi_fe_plus_fi_plus = self.update_TFs(fe + df, fi + df, Ue, Ui)
        TFe_fe_minus_fi_plus, TFi_fe_minus_fi_plus = self.update_TFs(fe - df, fi + df, Ue, Ui)
        TFe_fe_plus_fi_minus, TFi_fe_plus_fi_minus = self.update_TFs(fe + df, fi - df, Ue, Ui)
        TFe_fe_minus_fi_minus, TFi_fe_minus_fi_minus = self.update_TFs(fe - df, fi - df, Ue, Ui)
        
        #d2TFe/dfidfe
        self._diff2_TFe_fi_fe = (TFe_fe_plus_fi_plus - TFe_fe_minus_fi_plus -\
        TFe_fe_plus_fi_minus + TFe_fe_minus_fi_minus)/((2*df)**2)
        #d2TFi/dfidfe
        self._diff2_TFi_fi_fe = (TFi_fe_plus_fi_plus - TFi_fe_minus_fi_plus -\
        TFi_fe_plus_fi_minus + TFi_fe_minus_fi_minus)/((2*df)**2)
        #d2TFe/dfedfi
        self._diff2_TFe_fe_fi = self._diff2_TFe_fi_fe
        #d2TFi/dfedfi
        self._diff2_TFi_fe_fi = self._diff2_TFi_fi_fe
        
    def second_order(self, feIni, fiIni, external_input, TotTime, dt, df):
        """
        Second order integration of the MF equations
        
        feIni, fiIni: initial firing rates
        external_input: firing rate of the external drive
        TotTime: integration duration 
        dt: timestep
        df: step size for the derivation of the TF
        
        Returns arrays containing the time evolutions of
        -> Ue the recovery variable
        -> fe, fi the firing rates
        -> muge, mugi the average conductances
        -> stdge, stdgi the std of the conductances
        -> muve, muvi the average voltages
        -> stdve, stdvi the std of the voltages
        -> cee, cii the variances of the firing rates
        """
        self.df = df
        #Duration
        nSteps =  int(TotTime/dt)
        self.t2nd = np.linspace(0, TotTime, nSteps)
        
        #Turn external input into array if constant
        if type(external_input) == float or type(external_input) == int:
            external_input = np.ones(nSteps)*external_input
        
        #Initialize
        fecont = feIni; ficont = fiIni
        Ue = fecont*(self.dRS/self.aRS)
        Ui = 0
        
        #Covariances
        
        TFe, TFi = self.update_TFs(fecont + external_input[0], ficont, Ue, Ui)
        #cee = (TFe - fecont)*(TFe - fecont)
        #cie = (TFi - ficont)*(TFe - fecont)
        #cei = (TFi - ficont)*(TFe - fecont)
        #cii = (TFi - ficont)*(TFi - ficont)
        
        cee = 0
        cie = 0
        cei = 0
        cii = 0
        
        #Prepare lists
        self.LSUe2nd = np.empty(nSteps)
        
        #firing rates and variances
        self.LSfe2nd = np.empty(nSteps)
        self.LSfi2nd = np.empty(nSteps)
        self.LScee = np.empty(nSteps)
        self.LScie = np.empty(nSteps)
        self.LScei = np.empty(nSteps)
        self.LScii = np.empty(nSteps)
        
        #membrane conductances
        self.LSmuge2nd = np.empty(nSteps)
        self.LSmugi2nd = np.empty(nSteps)
        self.LSstdge2nd = np.empty(nSteps)
        self.LSstdgi2nd = np.empty(nSteps)
        
        #membrane potentials
        self.LSmuve2nd = np.empty(nSteps)
        self.LSmuvi2nd = np.empty(nSteps)
        self.LSstdve2nd = np.empty(nSteps)
        self.LSstdvi2nd = np.empty(nSteps)
        
        #Main loop
        for i in (range(len(self.t2nd))):
                
            #Store the previous step
            ceeold = cee; cieold = cie
            ceiold = cei; ciiold = cii
            
            #Update the population parameters with the current firing rates
            #self as they are needed to compute derivatives: sparsest way
            self.TFe, self.TFi = self.update_TFs(fecont+external_input[i], ficont, Ue, Ui, compute_stdg = True)
            
            #Store those values before they are modified for derivatives computations
            #Conductances
            self.LSmuge2nd[i] = self.muge
            self.LSmugi2nd[i] = self.mugi
            self.LSstdge2nd[i] = self.stdge
            self.LSstdgi2nd[i] = self.stdgi
            #Voltages
            self.LSmuve2nd[i] = self.muve
            self.LSmuvi2nd[i] = self.muvi
            self.LSstdve2nd[i] = self.stdve
            self.LSstdvi2nd[i] = self.stdvi
            
            #derivatives
            self.diffTF(fecont + external_input[i], ficont, Ue, Ui, df)
            
            #Update recovery value and covariances before updating firing rates
            Ue += dt*(-self.aRS*Ue + self.dRS*fecont)
            
            cee += dt/self.T * ((self.TFe - fecont)*(self.TFe - fecont)\
                                + 2*cee*self._diff_TFe_fe + 2*cei*self._diff_TFe_fi \
                                - 2*cee + ((self.TFe*((1/self.T) - self.TFe))/self.N2))
            if cee < 0:
                cee = 1e-9
                
            cei += dt/self.T * ((self.TFe - fecont)*(self.TFi - ficont)\
                                + cee*self._diff_TFi_fe + cie*self._diff_TFe_fe\
                                + cei*self._diff_TFi_fi + cii*self._diff_TFe_fi\
                                - 2*cei)
                
            cie += dt/self.T * ((self.TFe - fecont)*(self.TFi - ficont)\
                                + cei*self._diff_TFi_fi + cee*self._diff_TFi_fe\
                                + cie*self._diff_TFe_fe + cii*self._diff_TFe_fi\
                                - 2*cie)
                
            cii += dt/self.T * ((self.TFi - ficont)*(self.TFi - ficont)\
                                + 2*cie*self._diff_TFi_fe \
                                + 2*cii*self._diff_TFi_fi \
                                - 2*cii + ((self.TFi*((1/self.T) - self.TFi))/self.N1))
            if cii < 0:
                cii = 1e-9
                
            #Update firing rates (using old covariances values)
            fecont += dt/self.T * (self.TFe - fecont \
                                   + 0.5*ceeold*self._diff2_TFe_fe_fe\
                                   + 0.5*ceiold*self._diff2_TFe_fe_fi\
                                   + 0.5*cieold*self._diff2_TFe_fi_fe\
                                   + 0.5*ciiold*self._diff2_TFi_fe_fi)
            if fecont < 0:
                fecont = 0
                
            ficont += dt/self.T * (self.TFi - ficont \
                                   + 0.5*ceeold*self._diff2_TFi_fe_fe\
                                   + 0.5*ceiold*self._diff2_TFi_fi_fe\
                                   + 0.5*cieold*self._diff2_TFi_fi_fe\
                                   + 0.5*ciiold*self._diff2_TFi_fi_fi)
            if ficont < 0:
                ficont = 0
                
            #Store
            #Firing rates
            self.LSfe2nd[i] = fecont; self.LSfi2nd[i] = ficont
            #Var
            self.LScee[i] = cee; self.LScii[i] = cii
            self.LScie[i] = cie; self.LScei[i] = cei
            
            #Recovery
            self.LSUe2nd[i] = Ue
        
    def plots(self, order):
        
        if order == 'First order':
            
            plt.figure(dpi=600)
            plt.plot(self.t, self.LSfe, color = 'g')
            plt.plot(self.t, self.LSfi, color = '#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\nu$")
            plt.title("Firing rates vs time")
            plt.legend(["E", "I"])
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.t, self.LSmuge, color = 'g')
            plt.plot(self.t, self.LSmugi, color = '#CC3311')
            plt.fill_between(self.t, self.LSmuge - self.LSstdge,  self.LSmuge + self.LSstdge,
                         alpha=0.2, edgecolor='g', facecolor='g')
            plt.fill_between(self.t, self.LSmugi - self.LSstdgi,  self.LSmugi + self.LSstdgi,
                         alpha=0.2, edgecolor='#CC3311', facecolor='#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\mu_g$")
            plt.title("Conductances vs time")
            plt.legend(["E", "I"])
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.t, self.LSmuve, color = 'g')
            plt.plot(self.t, self.LSmuvi, color = '#CC3311')
            plt.fill_between(self.t, self.LSmuve - self.LSstdve,  self.LSmuve + self.LSstdve,
                         alpha=0.2, edgecolor='g', facecolor='g')
            plt.fill_between(self.t, self.LSmuvi - self.LSstdvi,  self.LSmuvi + self.LSstdvi,
                         alpha=0.2, edgecolor='#CC3311', facecolor='#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\mu_V$")
            plt.title("Voltages vs time")
            plt.legend(["E", "I"])
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.LSfe, self.LSfi)
            plt.xlabel(r"$\nu_e$")
            plt.ylabel(r"$\nu_i$")
            plt.title("System's trajectory in the phase space")
            plt.show()
            
        elif order == 'Second order':
            
            plt.figure(dpi=600)
            plt.plot(self.t2nd, self.LSfe2nd, color = 'g')
            plt.plot(self.t2nd, self.LSfi2nd, color = '#CC3311')
            plt.fill_between(self.t2nd, self.LSfe2nd - np.sqrt(self.LScee),  self.LSfe2nd + np.sqrt(self.LScee),
                         alpha=0.2, edgecolor='g', facecolor='g')
            plt.fill_between(self.t2nd, self.LSfi2nd - np.sqrt(self.LScii),  self.LSfi2nd + np.sqrt(self.LScii),
                         alpha=0.2, edgecolor='#CC3311', facecolor='#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\nu$")
            plt.title("Firing rates vs time - 2nd order: df = " + str(self.df))
            plt.legend(["E", "I"])
            #plt.ylim(0, 35)
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.t2nd, self.LSmuge2nd, color = 'g')
            plt.plot(self.t2nd, self.LSmugi2nd, color = '#CC3311')
            plt.fill_between(self.t2nd, self.LSmuge2nd - self.LSstdge2nd,  self.LSmuge2nd + self.LSstdge2nd,
                         alpha=0.2, edgecolor='g', facecolor='g')
            plt.fill_between(self.t2nd, self.LSmugi2nd - self.LSstdgi2nd,  self.LSmugi2nd + self.LSstdgi2nd,
                         alpha=0.2, edgecolor='#CC3311', facecolor='#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\mu_g$")
            plt.title("Conductances vs time - 2nd order")
            plt.legend(["E", "I"])
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.t2nd, self.LSmuve2nd, color = 'g')
            plt.plot(self.t2nd, self.LSmuvi2nd, color = '#CC3311')
            plt.fill_between(self.t2nd, self.LSmuve2nd - self.LSstdve2nd,  self.LSmuve2nd + self.LSstdve2nd,
                         alpha=0.2, edgecolor='g', facecolor='g')
            plt.fill_between(self.t2nd, self.LSmuvi2nd - self.LSstdvi2nd,  self.LSmuvi2nd + self.LSstdvi2nd,
                         alpha=0.2, edgecolor='#CC3311', facecolor='#CC3311')
            plt.xlabel("t")
            plt.ylabel(r"$\mu_V$")
            plt.title("Voltages vs time - 2nd order")
            plt.legend(["E", "I"])
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.t2nd, self.LScee)
            plt.plot(self.t2nd, self.LScie)
            plt.plot(self.t2nd, self.LScei)
            plt.plot(self.t2nd, self.LScii)
            plt.xlabel("t")
            plt.ylabel(r"$c$")
            plt.title("(Co)variances vs time - 2nd order")
            plt.legend(["$c_{ee}$", "$c_{ie}$", "$c_{ei}$", "$c_{ii}$"])
            plt.ylim(-1, 2)
            plt.show()
            
            plt.figure(dpi=600)
            plt.plot(self.LSfe2nd, self.LSfi2nd)
            plt.xlabel(r"$\nu_e$")
            plt.ylabel(r"$\nu_i$")
            plt.title("System's trajectory in the phase space - 2nd order")
            plt.show()
            
        else:
            raise Exception("Specify if 'First order' or 'Second order'")
