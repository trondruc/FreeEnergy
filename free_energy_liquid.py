#!/usr/bin/env python
#PBS -v PYTHONPATH
#PBS -e err.err
#PBS -o log.log

#############################################
#### Thermodynamic integration at
#### constant T and then constant rho
#### Assumes W/N and U/N as input
#### Produces Fex and Sex per particle
#############################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys, os, numpy, math

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import curve_fit

import rumd
from rumd.Simulation import Simulation
import rumd.Tools

if "PBS_O_WORKDIR" in os.environ:
    workingDir = os.environ["PBS_O_WORKDIR"]
    os.chdir(workingDir)
    sys.path.append(".")
    
class FreeEnergy(object):
    """ Class for implementing free energy calculations """
    def __init__(self, rhoMin=0.05, rhoMax=0.80, rhoStep=0.05,
                 startConfigFile="start.xyz.gz", LJcutoff=2.50):

        self.timeStep = 0.0025
        self.rhoMin = rhoMin
        self.rhoMax = rhoMax
        self.rhoStep = rhoStep
        self.potLJCutoff = LJcutoff

        self.TMin = 0.70
        self.TMax = 1.0
        self.TStep = 0.1
        
        self.n_equil_steps = 200000
        self.n_run_steps = 200000
        self.momentum_reset_interval = 200

        self.n_equil_stepsStart = 200000
        self.energyInterval = 50

        self.potLJ = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        self.potLJ.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=self.potLJCutoff);
        self.vLJ_cut = 4*(1/self.potLJCutoff**12 - 1/self.potLJCutoff**6)

        self.sim = Simulation(startConfigFile, 32, 8)
        self.sim.AddPotential(self.potLJ)
        self.T = None

    def Initialize(self):
        """ Create simulation object, integrator, potential, etc"""

        itg = rumd.IntegratorNVT(targetTemperature=self.T, timeStep=self.timeStep, thermostatRelaxationTime=0.2)
        self.sim.SetIntegrator(itg)

        self.SetDensity(self.rhoMin)

        self.sim.SetOutputScheduling("trajectory", "logarithmic")
        self.sim.SetOutputScheduling("energies", "linear", interval=self.energyInterval)
        self.sim.momentum_reset_interval = self.momentum_reset_interval

        self.sim.SetVerbose(False)

        self.sim.Run(self.n_equil_stepsStart, suppressAllOutput=True)

    def SetDensity(self, rh):
        """ scale simulation to get density rh"""
        nParticles = self.sim.GetNumberOfParticles()
        vol = self.sim.GetVolume()
        currentDensity = nParticles/vol
        scaleFactor = pow(rh/currentDensity, -1./3)

        self.sim.ScaleSystem(scaleFactor)

    def MainLoopRho(self, T):
        self.T = T
        nRuns = int((self.rhoMax - self.rhoMin)/self.rhoStep + 1.e-8) + 1

        self.Initialize()

        outfile = open("dataLJ_T_%.6f.dat" % T, "w")
        outfile.write("# rho, meanW, meanPE\n")

        for rdx in range(nRuns):

            rho = self.rhoMin + rdx*self.rhoStep
            print("rho, T", rho, T)
            self.SetDensity(rho)

            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()

            meanPE = meanVals["pe"]
            meanW = meanVals["W"]

            # Tail correction. Only for SCLJ.
            #tail_W = 1.0/rho * ((16.0/3.0)*math.pi*rho*rho*(2.0/3.0 * pow(1.0/self.potLJCutoff,9.0) - pow(1.0/self.potLJCutoff,3.0)))
            #meanW = meanW + tail_W

            #correction1 = (8.0/3.0)*math.pi*rho*((1.0/3.0)*pow(self.potLJCutoff,-9.0) - pow(self.potLJCutoff,-3.0))
            #correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut
            
            #tail_PE = correction1 + correction2
            #meanPE = meanPE + tail_PE
            
            outfile.write("%.6f %.6f %.6f\n" % (rho, meanW, meanPE))
        outfile.close()

    def MainLoopT(self, rho):
        nRuns = int((self.TMax - self.TMin)/self.TStep + 1.e-8) + 1

        outfile = open("dataLJ_rho_%.6f.dat" % rho, "w")
        outfile.write("# T, meanW, meanPE\n")

        for rdx in range(nRuns):

            temperature = self.TMax - rdx*self.TStep
            print("rho, T", rho, temperature)
            self.SetDensity(rho)
            self.sim.itg.SetTargetTemperature(temperature)

            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()
            meanPE = meanVals["pe"]
            meanW = meanVals["W"]

            # Tail correction. Only for SCLJ.
            #tail_W = 1.0/rho * ((16.0/3.0)*math.pi*rho*rho*(2.0/3.0 * pow(1.0/self.potLJCutoff,9.0) - pow(1.0/self.potLJCutoff,3.0)))

            #meanW = meanW + tail_W

            #correction1 = (8.0/3.0)*math.pi*rho*((1.0/3.0)*pow(self.potLJCutoff,-9.0) - pow(self.potLJCutoff,-3.0))
            #correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut
            
            #tail_PE = correction1 + correction2
            #meanPE = meanPE + tail_PE
            
            outfile.write("%.6f %.6f %.6f\n" % (temperature, meanW, meanPE))
        outfile.close()

# Polynomial fitting in log(rho). Virial expansion.
# The zero'th order is excluded as W = 0 at rho = 0.
def FitRho(rh, *params):
    r = 0.0
    e = 1

    for p in params:
        r += p*numpy.exp(e*rh)
        e += 1
    return r

# Rosenfeld-Tarazona fitting in 1/T.
# If the temperature range is very large, e.g. a decade or more, more terms should
# be added.
def FitTemperature(invT, a, b):
    r = a + b * numpy.power(invT, -3.0/5.0) 
    return r

if __name__ == "__main__":

    T = 2.100
    N = 1024
    NA = 1024
    NB = 0
    # Planck's constant for Argon in reduced units. Used in absolute F and S.
    h = 0.18574583293779054

    nParamsRhoScan = 6
    nParamsTScan = 2

    LJ = FreeEnergy()

    LJ.rhoMin = 0.05
    LJ.rhoMax = 0.85
    LJ.rhoStep = 0.05

    LJ.TMin = 0.70
    LJ.TMax = T
    LJ.TStep = 0.05

    print("Density scan...")
    LJ.MainLoopRho(T)

    print("Temperature scan...")
    LJ.MainLoopT(LJ.rhoMax)

    ####### Simulation done #######

    print("Calculation free energy...")

    # Read the generated data.
    infileRhoScan = open("dataLJ_T_%.6f.dat" % T)
    infileTScan = open("dataLJ_rho_%.6f.dat" % LJ.rhoMax)

    # Filter data.
    toFitListRhoScan = []

    nextLineRhoScan = infileRhoScan.readline()
    while nextLineRhoScan:
        while nextLineRhoScan.startswith("#"):
            nextLineRhoScan = infileRhoScan.readline()
        rho, meanW, meanPE = [float(item) for item in nextLineRhoScan.split()]
        toFitListRhoScan.append([math.log(rho), meanW, meanPE])

        nextLineRhoScan = infileRhoScan.readline()

    # Filter data.
    toFitListTScan = []
            
    nextLineTScan = infileTScan.readline()
    while nextLineTScan:
        while nextLineTScan.startswith("#"):
            nextLineTScan = infileTScan.readline()
        temperature, meanW, meanPE = [float(item) for item in nextLineTScan.split()]
        toFitListTScan.append([1/temperature, meanW, meanPE])

        nextLineTScan = infileTScan.readline()

    # Do fitting.
    toFitDataRhoScan = numpy.array(toFitListRhoScan)
    toFitDataTScan = numpy.array(toFitListTScan)
    
    fitParamsRhoScan, fitCovRhoScan = curve_fit(FitRho, toFitDataRhoScan[:, 0], toFitDataRhoScan[:, 1], numpy.zeros(nParamsRhoScan))
    fitParamsTScan, fitCovTScan = curve_fit(FitTemperature, toFitDataTScan[:, 0], toFitDataTScan[:, 2], numpy.zeros(nParamsTScan))
    
    ####### Calculate free energy #######
    
    # Mixing term for binary systems. 0 if SC.
    F_EX_Mix = - math.log(math.factorial(N)/(math.factorial(NA) * math.factorial(NB))) / N

    ############### Density scan #######################
    # W/V = - (dFex/dV) at constant T,N
    # => W = -(dFex/dlnV) => dFex = W dln(rho)
    ####################################################
    
    f_outfile = open("free_energy_T_%.6f.dat" % T, "w")
    f_outfile.write("# T FID/(NkBT) FEX/(NkBT) FTOT/(NkBT) SID/(NkB) SEX/(NkB) STOT/(NkB)\n")

    F_EX_Last = 0
    
    for rdx in range(len(toFitDataRhoScan)):
        rho = math.exp(toFitDataRhoScan[rdx,0])
        rhoStart = math.log(0.001)
        
        result = integrate.quad(FitRho, rhoStart, math.log(rho), args=tuple(fitParamsRhoScan))

        # Free energy over T in units of k_B
        F_DB = 3 * math.log( h / math.sqrt(2.0 * math.pi * T))
        F_ID = math.log(rho) - 1 + F_EX_Mix + F_DB 
        F_EX = result[0] / T
        F_TOT = F_ID + F_EX

        S_ID = -F_ID + 3.0/2.0
        S_EX = toFitDataRhoScan[rdx,2] / T - F_EX
        S_TOT = S_ID + S_EX

        f_outfile.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (rho, F_ID, F_EX, F_TOT, S_ID, S_EX, S_TOT))

        # Store the absolute free energy of the last density.
        if(rdx == (len(toFitDataRhoScan)) - 1):
            F_EX_Last = F_EX
                
    f_outfile.close()
   
    ############### Temperature scan #######################
    #  d(Fex/T)/d(1/T) = U at constant V,N
    # => d(Fex/T) = U d(1/T).
    ########################################################
    
    f_outfileRho = open("free_energy_rho_%.6f.dat" % LJ.rhoMax, "w")
    f_outfileRho.write("# T FID/(NkBT) FEX/(NkBT) FTOT/(NkBT) SID/(NkB) SEX/(NkB) STOT/(NkB)\n")

    for rdx in range(len(toFitDataTScan)):
        temperature = 1.0 / toFitDataTScan[rdx,0]

        # Gives Fex/T automatically.
        result = integrate.quad(FitTemperature, 1/T, 1/temperature, args=tuple(fitParamsTScan))

        # Free energy over T in units of k_B
        F_DB = 3 * math.log( h / math.sqrt(2.0 * math.pi * temperature))
        F_ID = (math.log(LJ.rhoMax) - 1) + F_EX_Mix + F_DB
        F_EX = result[0] + F_EX_Last
        F_TOT = F_ID + F_EX

        S_ID = -F_ID + 3.0/2.0
        S_EX = toFitDataTScan[rdx,2] / temperature - F_EX
        S_TOT = S_ID + S_EX
        
        f_outfileRho.write("%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (temperature, F_ID, F_EX, F_TOT, S_ID, S_EX, S_TOT))

    f_outfileRho.close()

    ##################### Plotting fits ####################

    f_plotRho = open("fitedRhoScan.dat", "w")
    
    for i in range(len(toFitDataRhoScan)):
        rho = math.exp(toFitDataRhoScan[i,0])
        value = FitRho(math.log(rho), *fitParamsRhoScan)
        f_plotRho.write("%.6f %.6f\n" % (rho, value))

    f_plotT = open("fitedTScan.dat", "w")
    
    for i in range(len(toFitDataTScan)):
        temperature = 1.0 / toFitDataTScan[i,0]
        value = FitTemperature(1.0 / temperature, *fitParamsTScan)
        f_plotT.write("%.6f %.6f\n" % (temperature, value))
