#!/usr/bin/env python3

# pass PYTHONPATH environment variable to the batch process
#PBS -v PYTHONPATH
#PBS -e error.txt
#PBS -o output.txt

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys, os, numpy, math

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.special as special

py_rumd_lib = {2:"../../../lib", 3:"../../../lib_py3"}[sys.version_info[0]]
sys.path.insert(1, py_rumd_lib)
sys.path.insert(1, "../../../Python")

import rumd
print("File used to import the rumd module", rumd.__file__)
from rumd.Simulation import Simulation
import rumd.Tools

if "PBS_O_WORKDIR" in os.environ:
    workingDir = os.environ["PBS_O_WORKDIR"]
    os.chdir(workingDir)
    sys.path.append(".")

lambda_values = 0.4 * numpy.array([0.2, 0.5, 1.0, 5.0, 10.0, 20.0, 40.516, 223.785, 598.683, 1255.95, 2358.18, 4177.98,5000, 7146.84, 11896.1, 19243.2, 30039.9,35000, 44796.8, 63089.6,70000, 82989.8, 100999, 115940.25])


class KA_FreeEnergy(object):
    """ Class for implementing free energy calculations """
    def __init__(self, rhoMin=1.2, rhoMax=1.20, rhoStep=0.05,
                 startConfigFile="start.xyz.gz", LJcutoff=2.50):

        self.timeStep = 0.0005
        self.rhoMin = rhoMin
        self.rhoMax = rhoMax
        self.rhoStep = rhoStep
        self.potLJCutoff = LJcutoff

        self.TMin = 0.1
        self.TMax = 1.0
        self.TStep = 0.1
        
        self.n_equil_steps = 5000000
        self.n_run_steps = 5000000
        self.momentum_reset_interval = 200

        self.n_equil_stepsStart = 5000000
        self.energyInterval = 50

        self.sim = Simulation(startConfigFile, 32, 8)

        self.potLJ = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        self.potLJ.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=self.potLJCutoff)
        self.potLJ.SetParams(i=0, j=1, Sigma=0.80, Epsilon=1.50, Rcut=self.potLJCutoff)
        self.potLJ.SetParams(i=1, j=0, Sigma=0.80, Epsilon=1.50, Rcut=self.potLJCutoff)
        self.potLJ.SetParams(i=1, j=1, Sigma=0.88, Epsilon=0.50, Rcut=self.potLJCutoff)
        self.vLJ_cut = 4*(1/self.potLJCutoff**12 - 1/self.potLJCutoff**6)
        
        self.sim.SetPotential(self.potLJ)

        self.ks = 57970.12499999999 * 2 * 0.4
        self.potWall = rumd.TetheredGroup(solidAtomTypes=[0,1], springConstant=self.ks)
        self.sim.AddPotential(self.potWall)
        
        self.T = None

    def Initialize(self):
        """ Create simulation object, integrator, potential, etc"""

        itg = rumd.IntegratorNPTLangevin(timeStep=self.timeStep, targetTemperature=self.T, friction=0.5, 
                                         targetPressure=0.1, barostatFriction=1.e-3, barostatMass=1.e-2);
        itg.SetBarostatMode(rumd.Off)
        #self.sim.sample.SetNumberOfDOFs(self.sim.GetNumberOfParticles()*3)  # TODO fix CM
        
        #itg = rumd.IntegratorNVT(targetTemperature=self.T, timeStep=self.timeStep, thermostatRelaxationTime=0.2)
        self.sim.SetIntegrator(itg)

        self.SetDensity(self.rhoMin)

        self.sim.SetOutputScheduling("trajectory", "logarithmic")
        self.sim.SetOutputScheduling("energies", "linear", interval=self.energyInterval)
        self.sim.SetOutputMetaData("energies", solid=True)
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

    def SetPotential(self, la):
        self.potWall.SetSpringConstant(la)
        
    def MainLoopRho(self, T):
        self.T = T
        nRuns = int((self.rhoMax - self.rhoMin)/self.rhoStep + 1.e-8) + 1

        self.Initialize()

        outfile = open("dataKA%.3f.dat" % T, "w")
        outfile.write("# lambda, meanW, meanPE\n")

        for rdx in range(0,len(lambda_values)):
            strength = lambda_values[rdx]

            print("rdx, strength", rdx, strength)
            self.SetPotential(strength)

            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()
            
            meanPE = meanVals["pe"]
            meanW = meanVals["W"]

            rho = 1.05
            correction1 = (8.0/3.0)*math.pi*rho*(- pow(self.potLJCutoff,-3.0))
            correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut
            meanPE = meanPE 
            
            meanSolid = meanVals["solid"] 

            outfile.write("%.5f %.5f %.5f\n" % (strength, meanW, meanSolid))
            outfile.flush()
        outfile.close()

    def MainLoopT(self, rho):
        nRuns = int((self.TMax - self.TMin)/self.TStep + 1.e-8) + 1

        outfile = open("dataKA%.3f.dat" % rho, "w")
        outfile.write("# T, meanW, meanPE\n")

        for rdx in range(nRuns):

            temperature = self.TMax - rdx*self.TStep
            print("rdx, T", rdx, temperature)
            self.SetDensity(rho)
            self.sim.itg.SetTargetTemperature(temperature)

            self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
            self.sim.Run(self.n_run_steps)

            rs = rumd.Tools.rumd_stats()
            rs.ComputeStats()

            meanVals = rs.GetMeanVals()
            meanPE = meanVals["pe"]
            meanW = meanVals["W"]

            rho = 1.05
            correction1 = (8.0/3.0)*math.pi*rho*( - pow(self.potLJCutoff,-3.0))
            correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut
            
            tail_PE = correction1 
            meanPE = meanPE 
            
            outfile.write("%.3f %.5f %.5f\n" % (temperature, meanW, meanPE))
        outfile.close()


def FitRho(rh, *params):
    r = 0.0
    e = 1

    for p in params:
        r += p*numpy.power(rh, e)
        e += 1
    return r

def FitTemperature(invT, a, b):
    r = a + b * numpy.power(invT,-3/5.0) 

    return r

if __name__ == "__main__":

    T = 0.48
    N = 12000
    NA = 8000
    NB = 4000

    # h in Argon reduced units.
    h = 0.18574583293779054

    nParams = 20
    nParamsRho = 2

    KA = KA_FreeEnergy()

    KA.rhoMin = 1.449119245433292

    KA.TMin = 0.75
    KA.TMax = T
    KA.TStep = 0.05
    
    c = numpy.exp(3.5)
    
    print("Run: Hamiltonian scan")
    #KA.MainLoopRho(T)

    print("Run: Temperature scan")
    #KA.MainLoopT(KA.rhoMin)
    
    # Do fitting of the data.
    infile = open("dataKA%.3f.dat" % T)
    infileRho = open("dataKA%.3f.dat" % KA.rhoMin)

    toFitList = []
    toFitListRho = []

    nextLine = infile.readline()

    # Filter out header.
    
    while nextLine:
        while nextLine.startswith("#"):
            nextLine = infile.readline()
        lambda_v, meanW, meanPE = [float(item) for item in nextLine.split()]
        toFitList.append([lambda_v, meanW, meanPE])

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
    
    fitParams, fitCov = curve_fit(FitRho, numpy.log(0.5*toFitData[:, 0]+c), (0.5*toFitData[:, 0]+c)*(2*toFitData[:, 2]/toFitData[:,0]), startParams)

    fitParamsRho, fitCovRho = curve_fit(FitTemperature, toFitDataRho[:, 0], toFitDataRho[:, 2], startParamsRho)
    
    # Mixing term.
    mix = math.factorial(N) // (math.factorial(NA) * math.factorial(NB))
    F_EX_Mix = -math.log(mix) / N 

    ############### Hamiltonian scan #######################
    
    f_outfile = open("free_energy_%.3f.dat" % T, "w")
    f_outfile.write("# F_ID F_E F_A2\n")

    beta = 1 / T
    V = N / KA.rhoMin

    # Free energy over T
    DB = h / math.sqrt(2.0 * math.pi * T)

    # Einstein crystal free energy.
    #F_E = (3.0/2.0) * (1 - 1/N) *  numpy.log( (0.5*KA.ks * beta * DB**2) / numpy.pi ) + (1/N) * numpy.log( N*DB**3/V ) - (3.0/(2.0*N)) * numpy.log(N)
    F_E = (3.0/2.0) * (1 - 1/N) *  numpy.log( (0.5*KA.ks * beta * DB**2) / numpy.pi ) + (1/N) * numpy.log( N*DB**3/V ) - 2.0 * numpy.log(N)/N
    #F_ID = math.log(KA.rhoMin) - 1 + F_EX_Mix + F_DB

    # Define function and interval
    a = numpy.log(c)
    b = numpy.log(c+KA.ks*0.5)

    # Gauss-Legendre (default interval is [-1, 1])
    deg = 20
    x, w = numpy.polynomial.legendre.leggauss(deg)

    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    gauss = sum(w * FitRho(t,tuple(fitParams))) * 0.5*(b - a)

    print(-gauss/T)
    
    for rdx in range(0,1):
        lambda_v = toFitData[rdx,0] * 0.5
        
        result = integrate.quad(FitRho, numpy.log(c), numpy.log(c+KA.ks*0.5), args=tuple(fitParams))

        f_outfile.write("%f %f\n" % (F_E, -result[0] / T))
                
    f_outfile.close()

    ############### Temperature scan #######################

    F = -6.098646871955663 + F_E - 7.693941
    #print(F)
    F_EX_Last = F 
    #print(F_EX_Last)
    
    f_outfileRho = open("free_energy_%.3f.dat" % KA.rhoMin, "w")
    f_outfileRho.write("# T F_ID F_EX F_TOT S_ID S_EX S_TOT\n")

    nRuns = int((KA.TMax - KA.TMin)/KA.TStep + 1.e-8) + 1
    
    for rdx in range(len(toFitDataRho)):
        temperature = 1.0 / toFitDataRho[rdx,0]
        
        result = integrate.quad(FitTemperature, 1/T, 1/temperature, args=tuple(fitParamsRho))

        # Free energy over T
        F_DB = 1#3 * math.log( h / math.sqrt(2.0 * math.pi * temperature))
        F_ID = (math.log(KA.rhoMax) - 1) + F_DB
        F_EX = result[0] + F_EX_Last
        F_TOT = F_ID + F_EX

        S_ID = 0 #-F_ID + 3.0/2.0
        S_EX = 0 #toFitDataRho[rdx,2] / temperature - F_EX
        S_TOT = 0 #S_ID + S_EX

        f_outfileRho.write("%f %f %f %f %f %f %f\n" % (temperature, F_ID, F_EX, F_TOT, S_ID, S_EX, S_TOT))

    f_outfileRho.close()
    
    ##################### Plotting fits ####################

    f_plot = open("plotL.dat", "w")

    for i in range(0,len(toFitData[:,0])):
        f_plot.write("%f %f\n" % (numpy.log(0.5*toFitData[i, 0]+c), (0.5*toFitData[i, 0]+c)*(2*toFitData[i, 2]/toFitData[i,0])))

    fitRhoR = []
    fitDataR = []

    for i in range(0,len(lambda_values)):
        newRho = numpy.log(0.5*toFitData[i,0] + c)
        fitRhoR.append(newRho)
        fitDataR.append(FitRho(newRho, *fitParams))

    f_plotN = open("plotLF.dat", "w")

    for i in range(len(fitRhoR)):
        f_plotN.write("%f %f\n" % (fitRhoR[i], fitDataR[i]))
