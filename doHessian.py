
import numpy as np
import sys
import gzip
import time
import math

np.set_printoptions(threshold=sys.maxsize)

class HesCalc:

    def __init__(self,T,state,rcut=2.5,write=False,fileName=0,nbList=0):

        self.pos = np.empty(0)
        self.img = np.empty(0)
        self.lBox = 0.
        self.nPart = 0
        
        if state == 0:

            if fileName == 0:
                raise HessianError("No data to analyze")
            self.readFile(fileName)

        else:
            self.pos=state[0]
            self.img=state[1]
            self.lBox=state[2]
            self.nPart=len(self.pos)
            assert(self.nPart==len(self.img))

        self.T = T
        self.writeAll = write
        self.hbar = 0.18574583293779054 / (2.0 * math.pi)

        self.epsilon = np.array([[1.00, 1.50],
                                 [1.50, 0.50]]) 
        
        self.sigma = np.array([[1.00, 0.80],
                               [0.80, 0.88]])
        
        self.rCutSq = rcut**2
        self.NA = int(math.ceil(self.nPart * 0.66666))

        self.dist = np.empty(0)
        self.distSq = np.empty(0)
        self.rSq = np.empty(0)
        self.rij = np.empty(0)
        self.idxArray = np.empty(0)

        print("Dist...")
        self.calcDist()
        print("NBList...")
        self.mNBList = nbList
        if self.mNBList == 0:
            self.calcNBList()

            
    # Read configuration from file.
    def readFile(self,file_Name):
        if file_Name[-2:] == 'gz':
            f_in = gzip.open(file_Name)
        else:
            f_in = open(file_Name)

        self.nPart = int(f_in.readline())
        info = f_in.readline()
        data = np.array([[float(ff) for ff in dd.split()] for dd in f_in.readlines()])
        f_in.close()

        t = np.array([int(ii) for ii in data[:,0]]) 
        self.pos = np.array([[ff for ff in dd] for dd in data[:,1:4]])
        #print self.pos
        self.img = np.array([[int(ii) for ii in dd] for dd in data[:,4:7]])
        #print self.img
        self.lBox = float(info.split()[5].split(',')[1])
        print self.lBox
        
    # Calculate particle distance
    def calcDist(self):
        # All pair distances.
        self.dist = self.pos - self.pos[:,np.newaxis]
        # Periodic image. Not using rint.
        self.dist += (self.dist < -self.lBox/2)*(self.lBox) 
        self.dist += (self.dist >  self.lBox/2)*(-self.lBox)
        self.distSq = self.dist**2
        # r^2 = dx^2 + dy^2 + dz^2
        self.rSq = np.sum(self.distSq,axis=2)
        # Distance r
        self.rij = np.sqrt(self.rSq) 
        
    # Calculate NB-list based on distances
    def calcNBList(self):
        rCutSqArr = np.ones_like(self.rSq) * self.rCutSq

        for i in range(0,self.nPart):
            for j in range(i+1,self.nPart):
                temp = self.sigma[i//self.NA, j//self.NA] * self.sigma[i//self.NA, j//self.NA]
                rCutSqArr[i,j] *= temp
                rCutSqArr[j,i] *= temp

        idx = np.where(self.rSq < rCutSqArr)
        self.idxArray = np.array(idx).T

        self.mNBList = np.zeros([self.nPart,self.nPart],dtype=int)

        for i,j in self.idxArray:
            self.mNBList[i,j] = 1

        # Remove self interaction.
        np.fill_diagonal(self.mNBList,0) 
        
    def doFullCalc(self):
        # The Hessian is a 3Nx3N dimensional array
        self.H = np.zeros([self.nPart*3,self.nPart*3]) 
        print("Calculation...")
        for i in range(0,self.nPart): 
            for j in range(i+1,self.nPart): 
                if (self.mNBList[i,j] == 1):
                    #print str(i) + str(" ") + str(j) + str(" ") + str(self.rij[i,j])
                    temp = self.offDiagNonBond(i,j)
                    self.H[3*i:3*(i+1),3*j:3*(j+1)] = temp 
                    self.H[3*j:3*(j+1),3*i:3*(i+1)] = temp
                    
            # Diagonal elements in H due to being partial i,i.
            for k in range(3):
                self.H[3*i:3*(i+1),3*i+k] = -np.sum(self.H[3*i:3*(i+1),k::3],axis=1) 

        print("Eigenvalue...")
        w = np.linalg.eigvalsh(self.H) 
        print("Eigenvalue done...")
        #w = np.sort(w)
        
        # Harmonic entropy.
        S_harm = 0
        for ww in w:
            if(ww > 1e-5):
                S_harm += (1 -  math.log(self.hbar * math.sqrt(ww) / self.T))

        
        f_out = open("EigVal.dat","w")
        for ww in w:
            f_out.write(str(ww)+"\n")

        f_out.close()

        #f_out = open("Hessian.dat","w")
        #for h in self.H:
        #    for hh in h:
        #        f_out.write(str(hh)+" ")
        #    f_out.write("\n")

        #f_out.close()
        print S_harm / self.nPart
        
        return S_harm / self.nPart

    
    # Calculating the contributions from non-bonded particles.
    def offDiagNonBond(self,n,m):
        dr = self.dist[n,m]
        h = dr * dr[:,np.newaxis]

        myRij = self.rij[n,m]
        myRijInv = 1./myRij

        v1 = self.LJfirst(myRijInv,n,m)
        u2 = (self.LJsecond(myRijInv,n,m) - v1 * myRijInv) * myRijInv**2
        u1 = v1 * myRijInv

        h *= u2
        # Diagonal term for 3x3 i,j has this term.
        h += np.identity(3) * u1 

        return -h

    
    # First and second order differentiated potentials.
    def LJfirst(self,rinv,i,j):
        s6 = self.sigma[i//self.NA,j//self.NA]**6
        
        return 4*self.epsilon[i//self.NA,j//self.NA]*(-12*s6**2*rinv**13 + 6*s6*rinv**7) 
    

    def LJsecond(self,rinv,i,j):
        s6 = self.sigma[i//self.NA,j//self.NA]**6

        return 4*self.epsilon[i//self.NA,j//self.NA]*(156*s6**2*rinv**14 - 42*s6*rinv**8)


