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
import rumd.analyze_energies as analyze

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

def getU0(potLJCutoff):
    """ Return energy when particles are on perfect lattice , U(R0) """
    sim = Simulation("start.xyz.gz")
    
    potLJ = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
    potLJ.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=potLJCutoff)
    potLJ.SetParams(i=0, j=1, Sigma=0.80, Epsilon=1.50, Rcut=potLJCutoff)
    potLJ.SetParams(i=1, j=0, Sigma=0.80, Epsilon=1.50, Rcut=potLJCutoff)
    potLJ.SetParams(i=1, j=1, Sigma=0.88, Epsilon=0.50, Rcut=potLJCutoff)
    
    sim.SetPotential(potLJ)
    sim.sample.CalcF()
    
    U0 = sim.sample.GetPotentialEnergy()   # The energy of the ideal crystal
    N = sim.sample.GetNumberOfParticles()
    u0 = U0/N
    del sim
    return u0

class KA_FreeEnergy(object):
    """ Class for implementing free energy calculations """
    def __init__(self, rhoMin=1.20, rhoMax=1.20, rhoStep=0.05,
                 startConfigFile="start.xyz.gz", LJcutoff=2.50):

        self.timeStep = 0.0005
        self.rhoMin = rhoMin
        self.rhoMax = rhoMax
        self.rhoStep = rhoStep
        self.potLJCutoff = LJcutoff

        self.TMin = 0.1
        self.TMax = 1.0
        self.TStep = 0.1
        
        self.n_equil_steps = 1000000
        self.n_run_steps = 5000000
        self.momentum_reset_interval = 200

        self.n_equil_stepsStart = 1000000
        self.energyInterval = 1

        self.sim = Simulation(startConfigFile, 32, 8)

        self.initialPositions = self.sim.sample.GetPositions()
        
        self.vLJ_cut = 4*(1/self.potLJCutoff**12 - 1/self.potLJCutoff**6)
	
        self.meanStatic = getU0(LJcutoff)
        print(self.meanStatic)

        rho = 1.05
        correction1 = (8.0/3.0)*math.pi*rho*( - pow(self.potLJCutoff,-3.0))
        correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut
            
        tail_PE = correction1 
        self.meanStatic = self.meanStatic
        
        #self.correction1 = (8.0/3.0)*math.pi*self.rhoMin*((1.0/3.0)*pow(self.potLJCutoff,-9.0) - pow(self.potLJCutoff,-3.0))
        
        #self.meanStatic = self.meanStatic 
        #print(self.meanStatic)
        
        self.ks = 115940.25 * 0.4
        self.potWall = rumd.TetheredGroup(solidAtomTypes=[0,1], springConstant=self.ks)
        self.sim.SetPotential(self.potWall)

        # Alternative calculator.
        self.potLJ2 = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        self.potLJ2.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=self.potLJCutoff)
        self.potLJ2.SetParams(i=0, j=1, Sigma=0.80, Epsilon=1.50, Rcut=self.potLJCutoff)
        self.potLJ2.SetParams(i=1, j=0, Sigma=0.80, Epsilon=1.50, Rcut=self.potLJCutoff)
        self.potLJ2.SetParams(i=1, j=1, Sigma=0.88, Epsilon=0.50, Rcut=self.potLJCutoff)

        self.alt_pot_calc = rumd.AlternatePotentialCalculator(sample=self.sim.sample, alt_pot=self.potLJ2)
        self.sim.AddExternalCalculator(self.alt_pot_calc)

        #self.sim.NewOutputManager("analysis")

        # set the output details
        #self.sim.SetOutputScheduling("analysis", "linear", interval=16)

        # add the callback function to the output manager
        #self.sim.RegisterCallback("analysis", HarmonicOsc, header="Harmonic")
        
        self.T = None

    def Initialize(self):
        """ Create simulation object, integrator, potential, etc"""

        self.itg = rumd.IntegratorNPTLangevin(timeStep=self.timeStep, targetTemperature=self.T, friction=0.5, 
                                              targetPressure=0.1, barostatFriction=1.e-3, barostatMass=1.e-2);
        self.itg.SetBarostatMode(rumd.Off)
        self.sim.SetIntegrator(self.itg)

        self.SetDensity(self.rhoMin)

        self.sim.SetOutputScheduling("trajectory", "logarithmic")
        self.sim.SetOutputScheduling("energies", "linear", interval=self.energyInterval)
        #self.sim.SetOutputMetaData("energies", solid=True)
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
        
        self.sim.Run(self.n_equil_steps, suppressAllOutput=True)
        self.sim.Run(self.n_run_steps)

        #print(self.itg.GetAcceptanceRate())
        
        nrgs = analyze.AnalyzeEnergies()
        nrgs.read_energies(['potLJ_12_6'])
        meanU = nrgs.energies['potLJ_12_6'] 

        rho = 1.05
        correction1 = (8.0/3.0)*math.pi*rho*( - pow(self.potLJCutoff,-3.0))
        correction2 = (2.0/3.0)*math.pi*rho*pow(self.potLJCutoff,3.0)*self.vLJ_cut

        tail_PE = correction1 
        meanU = meanU 

        print(self.meanStatic)
        print(meanU)
        
        potU = numpy.exp(-( (meanU - self.meanStatic)*N)/T)
        avgU = sum(potU) / len(potU)

        # free energy A / NT
        freeA = self.meanStatic / T  - (1.0/N) * numpy.log(avgU)
        
        print(avgU)
        print(freeA)
        
def FitRho(rh, *params):
    r = 0.0
    e = 0

    for p in params:
        r += p*numpy.power(rh, e)
        e += 1
    return r

def HarmonicOsc(sample):
    pos = sample.GetPositions()
    vis = sample.GetVelocities()

    L = pow( KA.sim.GetVolume(), 1.0/3.0 )
    
    diffX = pos[0,0] - KA.initialPositions[0,0]
    diffY = pos[0,1] - KA.initialPositions[0,1]
    diffZ = pos[0,2] - KA.initialPositions[0,2] 

    diffX = diffX - L * numpy.rint( diffX / L )
    diffY = diffY - L * numpy.rint( diffY / L )
    diffZ = diffZ - L * numpy.rint( diffZ / L )
    
    return "%.5f %.5f %.5f %.5f %.5f %.5f" % (diffX, diffY, diffZ, vis[0,0], vis[0,1], vis[0,2])

if __name__ == "__main__":

    T = 0.480
    N = 12000
    NA = 8000
    NB = 4000

    # h in Argon reduced units.
    h = 0.18574583293779054
    #h = 0.0635070*2*math.pi # Berthier
    nParams = 9

    KA = KA_FreeEnergy()

    KA.rhoMin = 1.449119245433292
    
    print("Run: Density scan")
    KA.MainLoopRho(T)
