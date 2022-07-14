#!/usr/bin/env python3

# pass PYTHONPATH environment variable to the batch process
#PBS -v PYTHONPATH
#PBS -e error.txt
#PBS -o output.txt

from __future__ import print_function, division, absolute_import

# this ensures PBS jobs run in the correct directory
import os, sys

import subprocess
import json

import rumd
from rumd.Simulation import Simulation
from rumd.Autotune import Autotune
import rumd.analyze_energies as analyze
import rumd.Tools

import numpy as np
import pandas
import matplotlib

import sys, os, numpy, math

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.special as special

W_ID = -0.6646666666666667

if 'PBS_O_WORKDIR' in os.environ:
    workingDir = os.environ["PBS_O_WORKDIR"]
    os.chdir(workingDir)
    sys.path.append(".")

    
class DB_FreeEnergy(object):
    """ Class for implementing free energy calculations """
    def __init__(self, rhoMin=0.05, rhoMax=0.80, rhoStep=0.05,
                 startConfigFile="start.xyz.gz", LJcutoff=2.50):

        self.timeStep = 0.001
        self.rhoMin = rhoMin
        self.rhoMax = rhoMax
        self.rhoStep = rhoStep
        self.potLJCutoff = LJcutoff

        self.TMin = 0.1
        self.TMax = 1.0
        self.TStep = 0.1
        
        self.n_equil_steps = 500000
        self.n_run_steps = 500000
        self.momentum_reset_interval = 200

        self.n_equil_stepsStart = 500000
        self.energyInterval = 20
        
	# create potential object. 
        self.potLJ = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedForce)
        self.potLJ.SetParams(i=0, j=0, Sigma=1.0, Epsilon=1.0, Rcut=2.50);

        self.sim = Simulation(startConfigFile, 32, 8)
        self.sim.AddPotential(self.potLJ)
        self.T = None

        # read topology file
        self.sim.ReadMoleculeData("start.top")

        self.cons_pot = rumd.ConstraintPotential()
        self.cons_pot.SetLinearConstraintMolecules(True)
        self.cons_pot.SetParams(bond_type=0, bond_length=1.000)
        self.sim.AddPotential(self.cons_pot)
        
    def Initialize(self):
        """ Create simulation object, integrator, potential, etc"""

        itg = rumd.IntegratorNVT(targetTemperature=self.T, timeStep=self.timeStep, thermostatRelaxationTime=0.2)
        self.sim.SetIntegrator(itg)

        self.SetDensity(self.rhoMin)

        self.sim.SetOutputScheduling("trajectory", "none")
        self.sim.SetOutputScheduling("energies", "linear", interval=self.energyInterval)
        self.sim.momentum_reset_interval = self.momentum_reset_interval

        self.sim.SetVerbose(False)

        self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())
        
        self.sim.Run(self.n_equil_stepsStart, suppressAllOutput=True)

        self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())

    def SetDensity(self, rh):
        """ scale simulation to get density rh"""
        nParticles = self.sim.GetNumberOfParticles() / 2.0
        vol = self.sim.GetVolume()
        currentDensity = nParticles/vol
        scaleFactor = pow(rh/currentDensity, -1./3)

        self.sim.ScaleSystem(scaleFactor, CM=True)

    def MainLoopRho(self, T):
        self.T = T
        nRuns = int((self.rhoMax - self.rhoMin)/self.rhoStep + 1.e-8) + 1

        self.Initialize()

        outfile = open("dataDB%.6f.dat" % T, "w")
        outfile.write("# rho, meanW, meanPE\n")

        for rdx in range(nRuns):

            rho = self.rhoMin + rdx*self.rhoStep
            print("rdx, rho", rdx, rho)
            self.SetDensity(rho)

            self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())
            
            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())
            
            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()

            meanPE = meanVals["pe"]
            meanW = meanVals["W"] 
            
            outfile.write("%.6f %.5f %.5f\n" % (rho, meanW, meanPE))
        outfile.close()

    def MainLoopT(self, rho):
        nRuns = int((self.TMax - self.TMin)/self.TStep + 1.e-8) + 1

        outfile = open("dataDB%.6f.dat" % rho, "w")
        outfile.write("# T, meanW, meanPE\n")

        for rdx in range(nRuns):

            temperature = self.TMax - rdx*self.TStep
            print("rdx, T", rdx, temperature)
            self.SetDensity(rho)
            self.sim.itg.SetTargetTemperature(temperature)

            self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())
            
            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            self.cons_pot.WritePotential(self.sim.sample.GetSimulationBox())
            
            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()
            meanPE = meanVals["pe"]
            meanW = meanVals["W"]
            
            outfile.write("%.6f %.5f %.5f\n" % (temperature, meanW, meanPE))
        outfile.close()

def FitRho(rh, *params):
    r = 0.0
    e = 1

    for p in params:
        r += p*numpy.power(numpy.exp(rh), e)
        e += 1
    return r

def FitTemperature(invT, a, b):
    r = a + b * numpy.power(invT,-3/5.0)

    return r

if __name__ == "__main__":

    T = 2.000
    N = 1000
    NA = 1000
    NB = 0

    # h in Argon reduced units.
    h = 0.18574583293779054
    nParams = 10
    nParamsRho = 2

    DB = DB_FreeEnergy()

    DB.rhoMin = 0.001
    DB.rhoMax = 0.55
    DB.rhoStep = 0.001

    DB.TMin = 1.700
    DB.TMax = 2.000
    DB.TStep = 0.1

    print("Run: Density scan")
    #DB.MainLoopRho(T)

    print("Run: Temperature scan")
    #DB.MainLoopT(DB.rhoMax)

    # Do fitting of the data.
    infile = open("dataDB%.6f.dat" % T)
    infileRho = open("dataDB%.6f.dat" % DB.rhoMax)

    toFitList = []
    toFitListRho = []
    
    nextLine = infile.readline()

    # Filter out header.
    
    while nextLine:
        while nextLine.startswith("#"):
            nextLine = infile.readline()
        rho, meanW, meanPE = [float(item) for item in nextLine.split()]
        toFitList.append([numpy.log(rho), meanW, meanPE])

        nextLine = infile.readline()

    # Filter out header.
        
    nextLineRho = infileRho.readline()

    while nextLineRho:
        while nextLineRho.startswith("#"):
            nextLineRho = infileRho.readline()
        temperature, meanW, meanPE = [float(item) for item in nextLineRho.split()]
        toFitListRho.append([1/temperature, meanW, meanPE])

        nextLineRho = infileRho.readline()

    startParams = numpy.zeros(nParams)
    toFitData = numpy.array(toFitList)

    startParamsRho = numpy.zeros(nParamsRho)
    toFitDataRho = numpy.array(toFitListRho)
    
    fitParams, fitCov = curve_fit(FitRho, toFitData[:, 0], 2*(toFitData[:, 1]-W_ID), startParams)
    
    fitParamsRho, fitCovRho = curve_fit(FitTemperature, toFitDataRho[:, 0], (1000*toFitDataRho[:, 2]/500.0), startParamsRho)
    
    # Mixing term.
    F_EX_Mix = - math.log(math.factorial(N)/(math.factorial(NA) * math.factorial(NB))) / N

    ############### Density scan #######################
    
    f_outfile = open("free_energy_%.6f.dat" % T, "w")
    f_outfile.write("# rho F_ID F_EX F_TOT S_ID S_EX S_TOT\n")

    F_EX_Last = 0
    
    for rdx in range(len(toFitData)):
        rho = numpy.exp(toFitData[rdx,0])

        result = integrate.quad(FitRho, numpy.log(0.0001), numpy.log(rho), args=tuple(fitParams))

        # Free energy over T
        F_DB = 3 * math.log(h / math.sqrt(2.0 * math.pi * T))
        F_ID = math.log(rho) - 1 + F_EX_Mix + F_DB 
        F_EX = result[0] / T
        F_TOT = F_ID + F_EX

        S_ID = -F_ID + 3.0/2.0
        S_EX = (((N*toFitData[rdx,2])/(N*0.5)) / T - F_EX)
        S_TOT = S_ID + S_EX
        
        f_outfile.write("%f %f %f %f %f %f %f\n" % (rho, F_ID, F_EX, F_TOT, S_ID, S_EX, S_TOT))

        if(rdx == len(toFitData) - 1):
            F_EX_Last = F_EX
                
    f_outfile.close()
   
    ############### Temperature scan #######################
    
    f_outfileRho = open("free_energy_%.6f.dat" % DB.rhoMax, "w")
    f_outfileRho.write("# T F_ID F_EX F_TOT S_ID S_EX S_TOT\n")

    nRuns = int((DB.TMax - DB.TMin)/DB.TStep + 1.e-8) + 1
    print(nRuns)
    for rdx in range(len(toFitDataRho)):
        temperature = 1.0 / toFitDataRho[rdx,0]
        
        result = integrate.quad(FitTemperature, 1/T, 1/temperature, args=tuple(fitParamsRho))

        # Free energy over T
        F_DB = 3 * math.log(h / math.sqrt(2.0 * math.pi * temperature))
        F_ID = (math.log(DB.rhoMax) - 1) + F_EX_Mix + F_DB
        F_EX = result[0] + F_EX_Last
        F_TOT = F_ID + F_EX

        S_ID = -F_ID + 3.0/2.0
        S_EX = 2*toFitDataRho[rdx,2] / temperature - F_EX
        S_TOT = S_ID + S_EX
        
        f_outfileRho.write("%f %f %f %f %f %f %f\n" % (temperature, F_ID, F_EX, F_TOT, S_ID, S_EX, S_TOT))

    f_outfileRho.close()

    ##################### Plotting fits ####################

    f_plot = open("plot.dat", "w")

    for i in range(len(toFitDataRho[:,0])):
        f_plot.write("%f %f\n" % (toFitDataRho[i,0], toFitDataRho[i,2]))

    fitRho = []
    fitData = []

    for i in range(int((DB.TMax-DB.TMin) / 0.001)):
        newT = 1 / ( DB.TMin + i*0.001 )
        fitRho.append(newT)
        fitData.append(FitTemperature(newT, *fitParamsRho))

    f_plotN = open("plotF.dat", "w")

    for i in range(len(fitRho)):
        f_plotN.write("%f %f\n" % (fitRho[i], fitData[i]))

    ##################### Plotting fits ####################

    f_plot = open("plotR.dat", "w")

    for i in range(len(toFitData[:,0])):
        f_plot.write("%f %f\n" % (numpy.exp(toFitData[i,0]), (toFitData[:, 1]-W_ID)[i]))

    fitRhoR = []
    fitDataR = []

    for i in range(int((DB.rhoMax-DB.rhoMin) / 0.0001)):
        newRho = ( DB.rhoMin + i*0.0001 )
        fitRhoR.append(newRho)
        fitDataR.append(FitRho(numpy.log(newRho), *fitParams))

    f_plotN = open("plotFR.dat", "w")

    for i in range(len(fitRhoR)):
        f_plotN.write("%f %f\n" % (fitRhoR[i], fitDataR[i]))
