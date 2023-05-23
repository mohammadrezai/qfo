# Quantum Fourier Optical Gate Optimization 

# The program calculates the optimal implementation of the desired gates in the quantum Fourier optics domain through various objective functions such as fidelity and success probability or a combination of them.
# Due to the nature of optimization programs, it is challenging to write a program that can systematically (in one run of the program) give the optimal variables for any desired gate.  
# Therefore, the user must actively modify objective functions, optimization parameters, and variables in several steps to get the optimal gate in a reasonable run time.

# Copyright (C) 2023, Mohammad Rezai
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numpy import linalg as LA
from scipy import optimize
from scipy.fft import fft, ifft
import random
import matplotlib.pyplot as plt
import sys, select
class sgdoptimize():
    def __init__(self,func,grad_func, start_point,num_iterations=1000,eta=0.1):
        thetat = np.array(start_point)
        print('g0',np.array(grad_func(thetat)))
        for t in range(num_iterations):
            gt = np.array(grad_func(thetat))
            thetat = thetat-eta*gt
            print('sgd=',func(thetat))
        print('grid1',np.array(grad_func(thetat)))
        self.fun=func(thetat)
        self.x=thetat

class adamoptimize():
    def __init__(self,func,grad_func, start_point,num_iterations=100000,alpha=0.0001,beta1=.9, beta2=.999,epsilon=1e-8):
        thetat = np.array(start_point)
        print('grid0',np.array(grad_func(thetat)))
        print('start, adam=',func(thetat))
        mt=0
        vt=0
        t=0
        functheta=10000
        while t < num_iterations:
            gt = np.array(grad_func(thetat))
            mt = beta1*mt + (1-beta1)*gt
            vt = beta2*vt + (1-beta2)*(gt**2)

            mt_hat = mt/(1-beta1**(t+1))
            vt_hat = vt/(1-beta2**(t+1))
            
            thetat = thetat - alpha*(mt_hat/(np.sqrt(vt_hat)+epsilon))
            functheta=func(thetat)
            print('to stop iteration return 1, adam=',functheta)
            i, o, e = select.select([sys.stdin], [], [], .01)
            if (i):
                input=sys.stdin.readline().strip()
                print('t=',t ,"the bounds are adjusted,", input)
                if input=='1':
                    t=num_iterations
            t+=1
        print('grid1',np.array(grad_func(thetat)))
        self.fun=func(thetat)
        self.x=thetat


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
np.set_printoptions(suppress=True)
Hm=np.array([
    [1,1],
    [1,-1]],dtype=np.float64)/(2**(1/2))

Hm2=np.array([
    [1,1,0,0],
    [1,-1,0,0],
    [0,0,1,1],
    [0,0,1,-1]],dtype=np.float64)/(2**(1/2))
iisH2=[2,1,-1,-2]
jjsH2=[2,1,-1,-2]
Hm3=np.array([
    [1,1,0,0,0,0],
    [1,-1,0,0,0,0],
    [0,0,1,1,0,0],
    [0,0,1,-1,0,0],
    [0,0,0,0,1,1],
    [0,0,0,0,1,-1]],dtype=np.float64)/(2**(1/2))
iisH3=[2,1,0,-1,-2,-3]
jjsH3=[2,1,0,-1,-2,-3]

qcx=np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
    ],dtype=np.float64)

r=1/(3**(1/2))
cxh=np.array([
    [r,0,0,0],
    [0,r,-1j*r,1j*r],
    [0,-1j*r,r,0],
    [0,1j*r,0,r]],dtype=complex)


cxm=cxh
iisCx=[1,0,-1,-2]#[0,1,2,3]
jjsCx=[1,0,-1,-2]#[0,1,2,3]
iisH=[1,0]
jjsH=[1,0]
class GateArray():
    def __init__(self,Gates,gatetype,Giis,Gjjs,Gshifts=[0],ijnormelG=[(0,0)]):
        self.size=len(Gates)
        self.ijnormelG=ijnormelG
        self.normelGs=np.empty(self.size,dtype=complex)
        for iG in range(self.size):
            if Gates[iG].shape[0]!=len(Giis[iG]) or Gates[iG].shape[1]!=len(Gjjs[iG]):
                print("ERROR")
                ex=1/0
            if Gates[iG][ijnormelG[iG][0],ijnormelG[iG][1]]==0:
                print('Error!')
                ex=1/0
            self.normelGs[iG]=Gates[iG][ijnormelG[iG][0],ijnormelG[iG][1]]
        
        self.Gates=Gates
        self.Giis=Giis
        self.Gjjs=Gjjs
        self.Gshifts=Gshifts
        self.gatetype=gatetype

        self.spgate=self.Sp(Gates)
        self.fidgate=abs(self.Fid(matarr=Gates,qpenalty=0,minfid=0,minsp=0,spfac=0))#self.Fid0(Gate)
        print('success probability=',self.spgate,'maximum fidality: Fid=',self.fidgate)
    
    def Sp(self,matarr): #success probability
        out0=0
        for mat in matarr:
            matH=mat.T.conj()
            out0+= abs(np.trace(matH @ mat)/(mat.shape[1]))
        return out0/len(matarr)

    def Sp0(self,mat): #success probability
        matH=mat.T.conj()
        return abs(np.trace(matH @ mat)/(mat.shape[1]))
    def Fid(self,matarr,qpenalty=0,minfid=0,minsp=0,spfac=0): #fidelity
        out0=0
        sp=0
        for iG in range(self.size):
            mat=matarr[iG]
            Gate=self.Gates[iG]
            GateH=Gate.T.conj()
            p=self.Sp0(mat)
            deno=p*Gate.shape[1]*Gate.shape[1]
            out0+=abs(((np.trace(GateH @ mat))**2)/deno)
            sp+=p
        sp/=len(matarr)
        out0/=len(matarr)
        out = out0 +.002*sp if sp >.111 else out0 +.1*p
        return -out0#+qpenalty


    def SetGdiff(self,matarr):
        self.dGmatarr=[]
        self.dGmatconjarr=[]
        self.coefmatatdGmat=[]
        for iG in range(self.size):
            mat=matarr[iG]
            Gate=self.Gates[iG]
            nelG=self.normelGs[iG]
            ielG=self.ijnormelG[iG][0]
            jelG=self.ijnormelG[iG][1]

            phaseamp=np.exp(-1j*np.angle(mat[ielG,jelG]/nelG))

            dmat=Gate-phaseamp*mat
            self.dGmatarr.append(dmat)
            self.dGmatconjarr.append(dmat.T.conj())
            self.coefmatatdGmat.append(-phaseamp)


    def Gdiff(self,qpenalty=0):
        out0=0
        for iG in range(self.size):
            out0+=np.sum(abs(self.dGmatarr[iG])**2)          
        return out0+qpenalty
    def Gdiff_derarr(self,matarr_derarr):
        derarr=[]
        for matarr in matarr_derarr:
            out0=0
            for iG in range(self.size):
                out0+=2*np.sum(np.real(self.coefmatatdGmat[iG]*matarr[iG]*self.dGmatconjarr[iG]))
            derarr.append(out0)
        return derarr        
      
class optimizer():
    floattype=float
    complextype=complex
    stop=0
    addmode=1
    addAs=2
    switchrandomly=4
    modifying=5
    printthesample=7
    zeroAs=[]
    def __init__(self,GA,nofAs0=1,nofmodes0=11,symetricG=True,invG=False,optfunc='diff',Asrangemin=-200,Asrangemax=200,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sin'):
        self.GA=GA
        self.spfac=0
        self.mindpenalty=1
        self.qpenaltydratio=.07
        self.qpenalfac=500

        self.spinat=0.991
        
        self.nofAs=nofAs0
        self.nofmodes=nofmodes0
        self.symetricG=symetricG
        self.invG=invG
        self.cosinv=-1
        
        self.setfunctions()
        self.setoptfunction(optfunc)
        self.thekind=thekind
        self.Asrangemin=Asrangemin
        self.Asrangemax=Asrangemax
        self.phisrangemin=phisrangemin
        self.phisrangemax=phisrangemax
        self.phifunctype=phifunctype

        if self.GA.gatetype == 'cxh':
            self.Gmat=self.qcxhmat
        elif self.GA.gatetype=='efarr':
            self.Gmat=self.EFmatarr
        else:
            1/0

    def expphifunc_der(self,As,k0x): # k0x =k0*x
        phi=np.zeros_like(k0x,dtype=self.floattype)
        if self.phifunctype=='sin':
            for i in range(As.size):
                phi+=As[i]*np.sin((i+1)*k0x)
            expphi=np.exp(-1.j*phi)
            derarr=[]
            for i in range(As.size):
                der_i=-1.j*np.sin((i+1)*k0x)*expphi
                derarr.append(der_i)
                
                
        elif self.phifunctype=='sincos':
            for i in range(As.size):
                if i%2!=0:
                    phi+=As[i]*np.cos((int(i/2)+1)*k0x)
                else:
                    phi+=As[i]*np.sin((int(i/2)+1)*k0x)
            expphi=np.exp(-1.j*phi)
            derarr=[]
            for i in range(As.size):
                if i%2!=0:
                    der_i=-1.j*np.cos((int(i/2)+1)*k0x)*expphi
                else:
                    der_i=-1.j*np.sin((int(i/2)+1)*k0x)*expphi
                derarr.append(der_i)

        elif self.phifunctype=='cos':
            for i in range(As.size):
                phi+=As[i]*np.cos((i+1)*k0x)

            expphi=np.exp(-1.j*phi)
            derarr=[]
            for i in range(As.size):
                der_i=-1.j*np.cos((i+1)*k0x)*expphi
                derarr.append(der_i)
        else:
            ex=1/0
        return derarr
        
    def phifunc(self,As,k0x): # k0x =k0*x
        phi=np.zeros_like(k0x,dtype=self.floattype)
        if self.phifunctype=='sin':
            for i in range(As.size):
                phi+=As[i]*np.sin((i+1)*k0x)
        elif self.phifunctype=='sincos':
            for i in range(As.size):
                if i%2!=0:
                    phi+=As[i]*np.cos((int(i/2)+1)*k0x)
                else:
                    phi+=As[i]* np.sin((int(i/2)+1)*k0x)
        elif self.phifunctype=='cos':
            for i in range(As.size):
                phi+=As[i]* np.sin((i+1)*k0x)
        else:
            ex=1/0

        return phi
    def qsofAs(self,As1,As2):
        k0x=np.arange(0,2*np.pi,2*np.pi/self.nofmodes,dtype=self.floattype)
        phi1=self.phifunc(As=As1,k0x=k0x)
        phi2=self.phifunc(As=As2,k0x=k0x)
        q1=fft(np.exp(-1.j*phi1))/self.nofmodes
        q2=fft(np.exp(-1.j*phi2))/self.nofmodes
        return q1,q2,phi1,phi2

    def qsofA(self,As1):
        k0x=np.arange(0,2*np.pi,2*np.pi/self.nofmodes,dtype=self.floattype)
        phi1=self.phifunc(As=As1,k0x=k0x)
        q1=fft(np.exp(-1.j*phi1))/self.nofmodes
        return q1,q1,phi1,phi1

    def qsofA_der(self,As1):
        k0x=np.arange(0,2*np.pi,2*np.pi/self.nofmodes,dtype=self.floattype)
        expphi_derarr=self.expphifunc_der(As=As1,k0x=k0x)
        q1_derarr=[]
        for expphi_der in expphi_derarr:
            q1_der=fft(expphi_der)/self.nofmodes
            q1_derarr.append(q1_der)
        return q1_derarr

    def getqsfromAsphi_der(self,Asphi):
        """The EF Hadamard function"""
        As1=Asphi[0:self.nofAs]
        if self.symetricG:
            if Asphi.size!= self.nofAs+self.nofmodes:
                print("ERROR1: Asphi.size!= nofAs+nofmodes ")
                ex=1/0
                sys.exit()
            phi=Asphi[self.nofAs:]
            if self.invG:
                As2=self.Asinv(As1)
                err=1/0
            else:
                q1_derarr=self.qsofA_der(As1)
        else:
            err=1/0
            if Asphi.size!= 2*self.nofAs+self.nofmodes:
                print("ERROR1: Asphi.size!= 2*nofAs+nofmodes ")
                ex=1/0
                sys.exit()
            As2=Asphi[self.nofAs:2*self.nofAs]
            phi=Asphi[2*self.nofAs:]
            q1,q2,phi1,phi2=self.qsofAs(As1,As2)
        return q1_derarr

    def getqsphisfromAsphi(self,Asphi):
        """The EF Hadamard function"""
        As1=Asphi[0:self.nofAs]
        if self.symetricG:
            if Asphi.size!= self.nofAs+self.nofmodes:
                print("ERROR1: Asphi.size!= nofAs+nofmodes ")
                ex=1/0
                sys.exit()
            phi=Asphi[self.nofAs:]
            if self.invG:
                As2=self.Asinv(As1)
                q1,q2,phi1,phi2=self.qsofAs(As1,As2)
            else:
                q1,q2,phi1,phi2=self.qsofA(As1)
        else:
            if Asphi.size!= 2*self.nofAs+self.nofmodes:
                print("ERROR1: Asphi.size!= 2*nofAs+nofmodes ")
                ex=1/0
                sys.exit()
            As2=Asphi[self.nofAs:2*self.nofAs]
            phi=Asphi[2*self.nofAs:]
            q1,q2,phi1,phi2=self.qsofAs(As1,As2)
        return q1,q2,phi1,phi2,phi


    def EFel(self,q1,phi,q2,n,l):
        out=0

        # actual q1 is q1 + some zero in the middle i.e. q[r]=0 for r>q.size/2 or r<-q.size/2
        qsize=q1.size
        mxr=min(int((qsize+1)/2),-n+int((qsize+1)/2),-l+int((qsize+1)/2)) #>0 because abs(n, l) < q.size/2
        mnr=max(-int((qsize)/2),-n-int((qsize)/2),-l-int((qsize)/2))#<0  because abs(n, l) < q.size/2
        for r in range(mnr,mxr,1):
            out+=q1[n+r]*np.exp(-1.j*phi[r])*q2[r+l]# =>: to avoid phi(r)->phi(-r) for simulation
        return out

    def EFel_derq(self,q1_der,q2_der,q1,phi,q2,n,l):
        if not self.symetricG:
            err=1/0
        # actual q1 is q1 + some zero in the middle i.e. q[r]=0 for r>q.size/2 or r<-q.size/2
        qsize=q1.size
        mxr=min(int((qsize+1)/2),-n+int((qsize+1)/2),-l+int((qsize+1)/2)) #>0 because abs(n, l) < q.size/2
        mnr=max(-int((qsize)/2),-n-int((qsize)/2),-l-int((qsize)/2))#<0  because abs(n, l) < q.size/2
        out=0
        for r in range(mnr,mxr,1):
            out+=q1_der[n+r]*np.exp(-1.j*phi[r])*q2[r+l]+q1[n+r]*np.exp(-1.j*phi[r])*q2_der[r+l]# =>: to avoid phi(r)->phi(-r) for simulation
        return out
    def EFel_derphi(self,q1,phi,r,q2,n,l):
        # actual q1 is q1 + some zero in the middle i.e. q[r]=0 for r>q.size/2 or r<-q.size/2
        qsize=q1.size
        mxr=min(int((qsize+1)/2),-n+int((qsize+1)/2),-l+int((qsize+1)/2)) #>0 because abs(n, l) < q.size/2
        mnr=max(-int((qsize)/2),-n-int((qsize)/2),-l-int((qsize)/2))#<0  because abs(n, l) < q.size/2
        if r<mxr:
            out=-1.j*q1[n+r]*np.exp(-1.j*phi[r])*q2[r+l]# =>: to avoid phi(r)->phi(-r) for simulation
        elif r< mnr+qsize:
            out=0
        else: 
            out=-1.j*q1[n+r-qsize]*np.exp(-1.j*phi[r])*q2[r+l-qsize]# =>: to avoid phi(r)->phi(-r) for simulation
        return out

    def EFel_derarr(self,q1_derarr,q2_derarr,q1,phi,q2,n,l):
                                                
        # actual q1 is q1 + some zero in the middle i.e. q[r]=0 for r>q.size/2 or r<-q.size/2
        qsize=q1.size
        mxr=min(int((qsize+1)/2),-n+int((qsize+1)/2),-l+int((qsize+1)/2)) #>0 because abs(n, l) < q.size/2
        mnr=max(-int((qsize)/2),-n-int((qsize)/2),-l-int((qsize)/2))#<0  because abs(n, l) < q.size/2
        outarr=[]
        for ider in range(self.nofAs):
            out=0
            for r in range(mnr,mxr,1):
                out+=q1_derarr[ider][n+r]*np.exp(-1.j*phi[r])*q2[r+l]+q1[n+r]*np.exp(-1.j*phi[r])*q2_derarr[ider][r+l]# =>: to avoid phi(r)->phi(-r) for simulation
            outarr.append(out)

        for r in range(0,mxr,1):
            out=-1.j*q1[n+r]*np.exp(-1.j*phi[r])*q2[r+l]# =>: to avoid phi(r)->phi(-r) for simulation
            outarr.append(out)
        for r in range(mxr,mnr+qsize,1):
            outarr.append(0)
        for r in range(mnr+qsize,qsize,1):
            #print('m',r)
            out=-1.j*q1[n+r-qsize]*np.exp(-1.j*phi[r])*q2[r+l-qsize]# =>: to avoid phi(r)->phi(-r) for simulation
            outarr.append(out)

        
    def EFphiel(self,n,l,phis=np.array([]),Asphi=np.array([])):
        """The EF Hadamard function"""
        if phis.size!=0:
            nofmodes=int(phis.size/3)       
            phisarr=phis.reshape((3,int(nofmodes)))
            q1=fft(np.exp(-1.j*phisarr[0]))/nofmodes
            phi=phisarr[1]
            q2=fft(np.exp(-1.j*phisarr[2]))/nofmodes
        else:
            q1,q2,phi1,phi2,phi= self.getqsphisfromAsphi(Asphi=Asphi)
        
        return self.EFel(q1,phi,q2,n,l)
    def givegatematof(self, Asphi):
        q1,q2,phi1,phi2,phi= self.getqsphisfromAsphi(Asphi=Asphi)
        return self.Gmat(q1=q1,q2=q2,phi=phi)
    def giveEFels(self,Asphi):
        out=np.empty([len(self.Giis),len(self.Gjjs)],dtype=self.complextype)
        for ii in range(self.Gate.shape[0]):
            for jj in range(self.Gate.shape[1]):
                out[ii,jj]=self.EFphiel(n=self.Giis[ii],l=self.Gjjs[jj],Asphi=Asphi)
        return out
    def qcxhmatofmat(self,out0):
        out=np.array([
            [out0[0,2]*out0[2,0]+out0[0,0]*out0[2,2],out0[0,2]*out0[3,0]+out0[0,0]*out0[3,2],out0[1,2]*out0[2,0]+out0[1,0]*out0[2,2],out0[1,2]*out0[3,0]+out0[1,0]*out0[3,2]],
            [out0[0,3]*out0[2,0]+out0[0,0]*out0[2,3],out0[0,3]*out0[3,0]+out0[0,0]*out0[3,3],out0[1,3]*out0[2,0]+out0[1,0]*out0[2,3],out0[1,3]*out0[3,0]+out0[1,0]*out0[3,3]], 
            [out0[0,2]*out0[2,1]+out0[0,1]*out0[2,2],out0[0,2]*out0[3,1]+out0[0,1]*out0[3,2],out0[1,2]*out0[2,1]+out0[1,1]*out0[2,2],out0[1,2]*out0[3,1]+out0[1,1]*out0[3,2]],
            [out0[0,3]*out0[2,1]+out0[0,1]*out0[2,3],out0[0,3]*out0[3,1]+out0[0,1]*out0[3,3],out0[1,3]* out0[2,1]+out0[1,1]*out0[2,3],out0[1,3]*out0[3,1]+out0[1,1]*out0[3,3]]])
        return out
    def qcxhmat(self,q1,q2,phi):
        out0=self.EFmat(q1,q2,phi)[0]
        out=np.array([
            [out0[0,2]*out0[2,0]+out0[0,0]*out0[2,2],out0[0,2]*out0[3,0]+out0[0,0]*out0[3,2],out0[1,2]*out0[2,0]+out0[1,0]*out0[2,2],out0[1,2]*out0[3,0]+out0[1,0]*out0[3,2]],
            [out0[0,3]*out0[2,0]+out0[0,0]*out0[2,3],out0[0,3]*out0[3,0]+out0[0,0]*out0[3,3],out0[1,3]*out0[2,0]+out0[1,0]*out0[2,3],out0[1,3]*out0[3,0]+out0[1,0]*out0[3,3]], 
            [out0[0,2]*out0[2,1]+out0[0,1]*out0[2,2],out0[0,2]*out0[3,1]+out0[0,1]*out0[3,2],out0[1,2]*out0[2,1]+out0[1,1]*out0[2,2],out0[1,2]*out0[3,1]+out0[1,1]*out0[3,2]],
            [out0[0,3]*out0[2,1]+out0[0,1]*out0[2,3],out0[0,3]*out0[3,1]+out0[0,1]*out0[3,3],out0[1,3]* out0[2,1]+out0[1,1]*out0[2,3],out0[1,3]*out0[3,1]+out0[1,1]*out0[3,3]]])

        return [out]
    def EFmat(self,q1,q2,phi):
        shi=self.GA.Gshifts[0]
        giis=self.GA.Giis[0]
        gjjs=self.GA.Gjjs[0]
        out=np.empty([len(giis), len(gjjs)],dtype=self.complextype)
        for ii in range(len(giis)):
            for jj in range(len(gjjs)):
                out[ii,jj]= self.EFel(q1,phi,q2,giis[ii]+shi,gjjs[jj]+shi)
        return [out]
    def EFmatarr_derarr(self,q1_derarr,q2_derarr,q1,q2,phi):
        
        if len(q1_derarr)!=len(q2_derarr) or len(q1_derarr)!=self.nofAs:
                err =1/0
        derarr=[]
        for ider in range(self.nofAs):
            q1_der=q1_derarr[ider]
            q2_der=q2_derarr[ider]
            matarr=[]
            for ig in range(self.GA.size):
                shi=self.GA.Gshifts[ig]
                giis=self.GA.Giis[ig]
                gjjs=self.GA.Gjjs[ig]
                out=np.empty([len(giis), len(gjjs)],dtype=self.complextype)
                for ii in range(len(giis)):
                    for jj in range(len(gjjs)):
                        out[ii,jj]= self.EFel_derq(q1_der=q1_der,q2_der=q2_der,q1=q1,phi=phi,q2=q2,n=giis[ii]+shi,l=gjjs[jj]+shi)
                matarr.append(out)
            derarr.append(matarr)

        for r in range(self.nofmodes):
            matarr=[]
            for ig in range(self.GA.size):
                shi=self.GA.Gshifts[ig]
                giis=self.GA.Giis[ig]
                gjjs=self.GA.Gjjs[ig]
                out=np.empty([len(giis), len(gjjs)],dtype=self.complextype)
                for ii in range(len(giis)):
                    for jj in range(len(gjjs)):
                        out[ii,jj]= self. EFel_derphi(q1=q1,phi=phi,r=r,q2=q2,n=giis[ii]+shi,l=gjjs[jj]+shi)
                matarr.append(out)
            derarr.append(matarr)
        return derarr

    def EFmatarr(self,q1,q2,phi):
        outarr=[]
        for ig in range(self.GA.size):
            shi=self.GA.Gshifts[ig]
            giis=self.GA.Giis[ig]
            gjjs=self.GA.Gjjs[ig]
            out=np.empty([len(giis), len(gjjs)],dtype=self.complextype)
            for ii in range(len(giis)):
                for jj in range(len(gjjs)):
                    out[ii,jj]= self.EFel(q1,phi,q2,giis[ii]+shi,gjjs[jj]+shi)
            outarr.append(out)
        return outarr
    ######################################################
    ##### optimization functions #########################
    ######################################################
    def EFGderphiqs(self,q1,q2,phi,q1_derarr,q2_derarr,optfun='diff'):
        mymat=self.Gmat(q1=q1,q2=q2,phi=phi)
        matarr_derarr=self.EFmatarr_derarr(q1_derarr=q1_derarr,q2_derarr=q2_derarr,q1=q1,q2=q2,phi=phi)

        if optfun=='diff':
            self.GA.SetGdiff(mymat)
            return self.GA.Gdiff_derarr(matarr_derarr) 

        else:
            print("ERROR: optfunc is not defined ")
            err=1/0
            sys.exit()

    def EFGoptphiqs(self,q1,q2,phi,optfun='diff'):
        mid=int(q1.size/2)
        dpenalty=max(self.mindpenalty,int(self.qpenaltydratio*mid))
        qpenalty0=np.sum(abs(q1[mid-dpenalty:mid+dpenalty]))
        qpenalty=self.qpenalfac*qpenalty0
        mymat=self.Gmat(q1=q1,q2=q2,phi=phi)

        if optfun=='diff':
            self.GA.SetGdiff(mymat)
            return self.GA.Gdiff(qpenalty=qpenalty) 
        elif optfun=='F':
            minfid=0
            minsp=0
            self.spfac=0
            return self.GA.Fid(matarr=mymat,qpenalty=qpenalty,minfid=minfid,minsp=minsp,spfac=self.spfac)
        elif optfun=='sp':
            return self.GA.Sp(mymat)
        else:
            print("ERROR: optfunc is not defined ")
            sys.exit()

    def EFGderAsphisym(self,Asphi,optfun):
        """The EF Hadamard function"""
        if Asphi.size!= self.nofAs+self.nofmodes:
            print("ERROR2: Asphi.size!= nofAs+nofmodes ",Asphi.size, self.nofAs, self.nofmodes)
            ex=1/0
            sys.exit()
        As1=Asphi[0:self.nofAs]
        phi=Asphi[self.nofAs:]  
        if self.invG:
            err=1/0
            As2=self.Asinv(As1)
            q1,q2,phi1,phi2= self.qsofAs(As1,As2)
        else:
            q1,q2,phi1,phi2=self.qsofA(As1)
            q_derarr=self.qsofA_der(As1)
         
        return self.EFGderphiqs(q1=q1,q2=q2,phi=phi,q1_derarr=q_derarr,q2_derarr=q_derarr,optfun=optfun)
    def EFGderAsphisymdiff(self,Asphi):
        return self.EFGderAsphisym(Asphi,optfun='diff') 
    def EFGoptAsphisym(self,Asphi,optfun):
        """The EF Hadamard function"""
        if Asphi.size!= self.nofAs+self.nofmodes:
            print("ERROR2: Asphi.size!= nofAs+nofmodes ",Asphi.size, self.nofAs, self.nofmodes)
            ex=1/0
            sys.exit()
        As1=Asphi[0:self.nofAs]
        phi=Asphi[self.nofAs:]  
        if self.invG:
            As2=self.Asinv(As1)
            q1,q2,phi1,phi2= self.qsofAs(As1,As2)
        else:
            q1,q2,phi1,phi2=self.qsofA(As1)
         
        
        
        return self.EFGoptphiqs(q1,q2,phi,optfun=optfun)

    def EFGoptAsphisymdiff(self,Asphi):
        return self.EFGoptAsphisym(Asphi,optfun='diff')    

    def EFGoptAsphisymF(self,Asphi):
        return self.EFGoptAsphisym(Asphi,optfun='F')  


    def EFGoptAsphisymsp(self,Asphi):
        return self.EFGoptAsphisym(Asphi,optfun='sp')  

    def EFGoptAsphi(self,Asphi,optfun):
        """The EF Hadamard function"""
        if Asphi.size!= 2*self.nofAs+self.nofmodes:
            m=1/0
            print("ERROR3: Asphi.size!= 2*nofAs+nofmodes ",Asphi.size, self.nofAs,self.nofmodes)
            sys.exit()
        As1=Asphi[0:self.nofAs]
        As2=Asphi[self.nofAs:2*self.nofAs]
        phi=Asphi[2*self.nofAs:]
        q1,q2,phi1,phi2= self.qsofAs(As1,As2)
        return self.EFGoptphiqs(q1,q2,phi,optfun=optfun)

    def EFGoptAsphidiff(self,Asphi):
        return self.EFGoptAsphi(Asphi,optfun='diff')
        
    def EFGoptAsphiF(self,Asphi):
        return self.EFGoptAsphi(Asphi,optfun='F')

    def EFGoptAsphisp(self,Asphi):
        return self.EFGoptAsphi(Asphi,optfun='sp')

    def EFGoptphi(self,phis,optfun):
        """The EF Hadamard function"""
        nofmodes=int(phis.size/3)
        phisarr=phis.reshape((3,nofmodes))
        q1=fft(np.exp(-1.j*phisarr[0]))/nofmodes
        phi=phisarr[1]
        q2=fft(np.exp(-1.j*phisarr[2]))/nofmodes
        return self.EFGoptphiqs(q1,q2,phi,optfun=optfun)

    def EFGoptphidiff(self,phis):
        return self.EFGoptphi(phis,optfun='diff')   
    def EFGoptphiF(self,phis):
        return self.EFGoptphi(phis,optfun='F')     
    
        
    def extendphiA(self,sample,nofaddedmode=0,nofaddedAs=0,thevalue=0):
        """extend phi"""
        #A0=random.random() # the initial guess for As
        addedAbound=(-1,1)
        addedphibound=(-np.pi, np.pi)

        if self.thekind=='phis':
            phis=sample
            #nofmodes=int(phis.size/3)
            newmode=self.nofmodes+nofaddedmode

            phisarr=phis.reshape((3,self.nofmodes))
            q1=fft(np.exp(-1.j*phisarr[0]))/self.nofmodes
            q2=fft(np.exp(-1.j*phisarr[2]))/self.nofmodes

            k0=1
            ks=np.array([k0*i if i <= int(self.nofmodes/2) else k0*(i-self.nofmodes) for i in range(self.nofmodes)])
            xs = np.arange(0,2*np.pi,2*np.pi/newmode)
            mypupil1=np.array([np.sum(q1*np.exp(1j*ks*x)) for x in xs])
            mypupil2=np.array([np.sum(q2*np.exp(1j*ks*x)) for x in xs])
            myphi1=-np.angle(mypupil1)
            myphi2=-np.angle(mypupil2)   
        
            phi=phisarr[1]
            addedphis=[np.pi*(-1+2*random.random()) if thevalue==0 else thevalue for i in range(nofaddedmode)]
            myphi=np.concatenate((phi[:int((self.nofmodes+1)/2)],addedphis,phi[int((self.nofmodes+1)/2):]), axis=0)
        
            outsample=np.concatenate((myphi1,myphi,myphi2), axis=0)
            ############################
            ############################
            phibounds = [(-np.pi, np.pi) for i in range(newmode)]
            phi1bounds = [(-np.pi, np.pi)for i in range(newmode)]
            phi2bounds = [(-np.pi, np.pi)for i in range(newmode)]
            outbounds=np.concatenate((phi1bounds,phibounds,phi2bounds), axis=0)
        elif self.thekind=='Asphi':
            Asphi=sample
            if self.symetricG:
                if Asphi.size!= self.nofAs+self.nofmodes:
                    print("ERROR40: Asphi.size!= nofAs+nofmodes ",Asphi.size,self.nofAs,self.nofmodes)
                    ex=1/0
                    sys.exit()
                As1=Asphi[0:self.nofAs]
                phi=Asphi[self.nofAs:]
                phibounds =[(-np.pi, np.pi) for i in range(len(phi))]
                As1bounds =[(As1[i]+self.Asrangemin, As1[i]+self.Asrangemax) for i in range(self.nofAs)]
                if nofaddedmode !=0:
                    print('nofaddedmode',nofaddedmode,phi.size)
                    addedphis=[np.pi*(-1+2*random.random()) if thevalue==0 else thevalue for i in range(nofaddedmode)]
                    myphi=np.concatenate((phi[:int((self.nofmodes+1)/2)],addedphis,phi[int((self.nofmodes+1)/2):]), axis=0)
                    addedphibounds=[addedphibound for i in range(nofaddedmode)]
                    myphibounds=np.concatenate((phibounds[:int((self.nofmodes+1)/2)],addedphibounds,phibounds[int((self.nofmodes+1)/2):]), axis=0)
                    print('nofaddedmode',nofaddedmode,myphi.size) 
                else:
                    myphi=phi
                    myphibounds=phibounds

                if nofaddedAs !=0:
                    addeda=0
                    A0s= [addeda for i in range(nofaddedAs)]
                    A0sbounds= [addedAbound for i in range(nofaddedAs)]
                    myAs1=np.concatenate((As1,A0s), axis=0)
                    myAs1bounds=np.concatenate((As1bounds,A0sbounds), axis=0)
                else:
                    myAs1= As1
                    myAs1bounds=As1bounds
                for i in self.zeroAs:
                    if i<len(myAs1bounds):
                        myAs1bounds[i]=(0,0)
                outsample=np.concatenate((myAs1,myphi), axis=0)
                outbounds=np.concatenate((myAs1bounds,myphibounds), axis=0)
                    
            else:
                if Asphi.size!= 2*self.nofAs+self.nofmodes:
                    print("ERROR41: Asphi.size!= 2*nofAs+nofmodes ")
                    sys.exit()
                As1=Asphi[0:self.nofAs]
                As2=Asphi[self.nofAs:2*self.nofAs]
                phi=Asphi[2*self.nofAs:]

                phibounds = [(-np.pi, np.pi) for i in range(len(phi))]
                As1bounds = [(As1[i]+self.Asrangemin, As1[i]+self.Asrangemax) for i in range(self.nofAs)]
                As2bounds = [(As2[i]+self.Asrangemin, As2[i]+self.Asrangemax) for i in range(self.nofAs)]                
                print('nofaddedmode',nofaddedmode,phi.size)                
                addedphi=(phi[int(self.nofmodes/2)]+phi[int(self.nofmodes/2)+1])/2
                addedphis=[addedphi for i in range(nofaddedmode)]
                myphi=phi if nofaddedmode==0 else np.concatenate((phi[:int((self.nofmodes+1)/2)],addedphis,phi[int((self.nofmodes+1)/2):]), axis=0)
                addedphibounds=[addedphibound for i in range(nofaddedmode)]
                
                myphibounds= phibounds if nofaddedmode==0 else np.concatenate((phibounds[:int((self.nofmodes+1)/2)],addedphibounds,phibounds[int((self.nofmodes+1)/2):]), axis=0)

                print('nofaddedmode',nofaddedmode,myphi.size) 
                if nofaddedAs !=0:
                    A0s= [random.random() for i in range(nofaddedAs)]
                    A0sbounds= [addedAbound for i in range(nofaddedAs)]
                    myAs1=np.concatenate((As1,A0s), axis=0) 
                    myAs2=np.concatenate((As2,A0s), axis=0) 
                    myAs1bounds=np.concatenate((As1bounds,A0sbounds), axis=0) 
                    myAs2bounds=np.concatenate((As2bounds,A0sbounds), axis=0) 
                else:   
                    myAs1=As1
                    myAs2=As2
                    myAs1bounds=As1bounds
                    myAs2bounds=As2bounds
                for i in self.zeroAs:
                    if i<len(myAs1bounds):
                        myAs1bounds[i]=(0,0)
                        myAs2bounds[i]=(0,0)
                outsample=np.concatenate((myAs1,myAs2,myphi), axis=0)
                outbounds=np.concatenate((myAs1bounds,myAs2bounds,myphibounds), axis=0)

        self.nofmodes+=nofaddedmode
        self.nofAs+=nofaddedAs
        return outsample, outbounds


    def plotEF(self,thedata):
        if self.thekind=='phis':
            phis=thedata
            nofmodes=int(phis.size/3)       
            phisarr=phis.reshape((3,int(nofmodes)))
            phi1=phisarr[0]
            phi=phisarr[1]
            phi2=phisarr[2]
            q1=fft(np.exp(-1.j*phisarr[0]))/nofmodes
            phi=phisarr[1]
            q2=fft(np.exp(-1.j*phisarr[2]))/nofmodes
        elif self.thekind=='Asphi':
            Asphi=thedata
            q1,q2,phi1,phi2,phi= self.getqsphisfromAsphi(Asphi=Asphi)
            nofmodes=phi.size

        plt.figure()
        plt.subplot(311)
        ph=np.arange(-np.pi,np.pi,2*np.pi/nofmodes)
        plt.plot(ph, np.roll(phi1,int(nofmodes/2)), 'r--',ph, np.roll(phi2,int(nofmodes/2)))
        plt.title(f'phi 1,2')
        
        plt.subplot(312)
        ks=np.array([-int(nofmodes/2)+ i for i in range(nofmodes)])
        plt.plot(ks, np.roll(np.abs(q1),int(nofmodes/2)), 'r--',ks, np.roll(np.abs(q2),int(nofmodes/2)))
        plt.title('qs')
        
        plt.subplot(313)
        ks=np.array([-int(nofmodes/2)+ i for i in range(nofmodes)])
        plt.plot(ks, np.roll(phi,int(nofmodes/2)), 'r--')
        plt.title('phi')
        plt.show()

    def Asinv(self,As):
        #self.cosinv=1 #-1
        out=np.copy(As)
        if self.phifunctype=='sin':
            pass
        elif self.phifunctype =='sincos':
            coef=self.cosinv
            for i in range(As.size):
                coef*=self.cosinv
                out[i]*=coef
        elif self.phifunctype =='cos':
            out[i]*=self.cosinv
        else:
            ex=1/0
            
        return out

    def makerandmatrix(self,nofas=1,symetric=True,Asrange=30):
        As1=np.array([Asrange*(1-2*random.random()) for i in range(nofas)])
        As2=self.Asinv(As1) if symetric else np.array([Asrange*(1-2*random.random()) for i in range(nofas)])
        norm=0
        nofm=7
        while norm <1.000000000000001:
            q1,q2,phi1,phi2=self.qsofAs(As1,As2)
            norm1= np.sum(abs(q1)**2)
            norm2= np.sum(abs(q2)**2)
            norm=(norm1+norm2)/2
            nofm+=6
            print(nofm,'nofmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm',norm,norm1,norm2)
        phi=np.array([np.pi*(-1+2*random.random()) for i in range(nofm)])

        Asphi=np.concatenate((As1,As2,phi), axis=None)
        Asphisym=np.concatenate((As1,phi), axis=None)
        if symetric: 
            myAsphi=Asphisym
        else:
            myAsphi=Asphi
        self.plotEF(thedata=myAsphi)
        Gmatrix=self.givegatematof(Asphi=myAsphi)
        print(bcolors.OKBLUE +"EFels="+ bcolors.ENDC,Gmatrix)
        return Gmatrix, myAsphi,nofm,nofas
    def setfunctions(self):
        if self.symetricG:
            self.difffunc=self.EFGoptAsphisymdiff
            self.diffderfunc=self.EFGderAsphisymdiff
            self.Ffunc=self.EFGoptAsphisymF
            self.spfunc=self.EFGoptAsphisymsp
        else:
            self.difffunc=self.EFGoptAsphidiff
            self.Ffunc=self.EFGoptAsphiF
            self.spfunc=self.EFGoptAsphisp
    def theNderfunc(self,Asphi):
        epsilon=1e-8
        myfunc0=self.theoptfunc(Asphi)
        out=np.empty(Asphi.size,dtype=self.floattype)
        for ii in range(Asphi.size):
            Asphi[ii]+=epsilon
            out[ii]=(self.theoptfunc(Asphi)-myfunc0)/epsilon
            Asphi[ii]-=epsilon

        return out
    def setoptfunction(self,optfunc):
        self.optfunc=optfunc
        if optfunc=='diff':
            self.theoptfunc= self.difffunc
            self.thederfunc=None
            self.thederfunc= self.theNderfunc
        elif optfunc=='F':      
            self.theoptfunc=self.Ffunc
            self.thederfunc= self.theNderfunc
    def givebounds(self):
        phibounds = [(self.phisrangemin, self.phisrangemin) for i in range(self.nofmodes)]
        if self.thekind=='phis':
            phi1bounds = [(self.phisrangemin, self.phisrangemax)for i in range(self.nofmodes)]
            phi2bounds = [(self.phisrangemin, self.phisrangemax)for i in range(self.nofmodes)]
            bounds=np.concatenate((phi1bounds,phibounds,phi2bounds), axis=0)
        elif self.thekind=='Asphi':
            As1bounds = [(self.Asrangemin, self.Asrangemax) for i in range(self.nofAs)]
            As2bounds = [(self.Asrangemin, self.Asrangemax) for i in range(self.nofAs)]
            for i in self.zeroAs:
                if i<len(As1bounds):
                    As1bounds[i]=(0,0)
                    As2bounds[i]=(0,0)
    
            if self.symetricG:
                bounds=np.concatenate((As1bounds,phibounds), axis=0)
            else:
                bounds=np.concatenate((As1bounds,As2bounds,phibounds), axis=0)
        return bounds
    
    def setcurrentsample(self,symetricG,phi,As1,As2=0,samplephifunctype='sin'):
        self.nofmodes=phi.size
        if samplephifunctype=='sin'and self.phifunctype=='sincos':
            as1=np.empty(2*As1.size,dtype=self.floattype)
            if not symetricG:
                as2=np.empty(2*As2.size,dtype=self.floattype)
            for i in range(As1.size):
                as1[2*i]=As1[i]
                as1[2*i+1]=0
                if not symetricG:
                    as2[2*i]=As2[i]
                    as2[2*i+1]=0
        elif samplephifunctype==self.phifunctype:
            as1=As1
            as2=As2
        else:
            print(samplephifunctype,'!=',self.phifunctype)
            1/0

        if self.symetricG:
            self.nofAs=as1.size
            self.currentsample=np.concatenate((as1,phi), axis=None)
        else:
            if as1.size==as2.size:
                self.nofAs=as1.size
                self.currentsample=np.concatenate((as1,as2,phi), axis=None)
            else:
                1/0

    def giverandsample(self):
        phi=np.array([np.pi*(-1+2*random.random()) for i in range(self.nofmodes)])
        if self.thekind=='phis':
            phi1=np.array([np.pi*(-1+2*random.random()) for i in range(self.nofmodes)])
            phi2=np.array([np.pi*(-1+2*random.random()) for i in range(self.nofmodes)])
            sample=np.concatenate((phi1,phi,phi2), axis=None)
        elif self.thekind=='Asphi':
            As1=np.array([-1+2* random.random() for i in range(self.nofAs)])
            if self.symetricG:
                sample=np.concatenate((As1,phi), axis=None)
            else:
                As2=np.array([-1+2* random.random() for i in range(self.nofAs)])
                sample=np.concatenate((As1,As2,phi), axis=None)
        return sample

    def switchsamplerand(self,nofrandswitches,sample):
        if self.thekind=='phis':
            for i in range(nofrandswitches):
                sample[random.randrange(sample.size)]=np.pi*(-1+2*random.random())
        elif self.thekind=='Asphi':
            Asrange=self.Asrangemax-self.Asrangemin
            for i in range(nofrandswitches):
                indrand=random.randrange(sample.size)
                if (self.symetricG and indrand<self.nofAs) or (not self.symetricG and indrand<2*self.nofAs):
                    sample[indrand]+=self.Asrangemin+Asrange* random.random()
                elif indrand+1<sample.size: #else:
                    sample[indrand]=(sample[indrand-1]+sample[indrand+1])/2
        return sample
    def globopt(self):
        res=optimize.differential_evolution(self.theoptfunc, bounds=self.givebounds())
        self.plotEF(thedata=res.x)
        print('diff=',self.difffunc(res.x)  ,'Fid=',self.Ffunc(res.x))
        print(bcolors.OKBLUE +"EFels="+ bcolors.ENDC,self.givegatematof(Asphi=res.x))
        self.currentsample=res.x
    def opt(self,method,nofmodestoadd=2,nofrandswitches=3, nofrands=20000000000000000000,askorder=True,usecurrentsample=False):
        order=1
        preres=500
        modify=False
        nofrandsind=nofrands
        if usecurrentsample:
            thesample=self.currentsample
            thesample,thebounds=self.extendphiA(sample=thesample,nofaddedmode=0,nofaddedAs=0)
        else:
            thesample=self.giverandsample()
            thebounds=self.givebounds()
        while order !=0:
            if method=='de':
                print('optimize.differential_evolution')
                res=optimize.differential_evolution(self.theoptfunc, bounds=thebounds,x0=thesample.flatten())
            elif method=='da':
                print('optimize.dual_annealing')
                res=optimize.dual_annealing(self.theoptfunc, bounds=thebounds,x0=thesample.flatten())
            elif method=='sgd':
                res=sgdoptimize(func=self.theoptfunc,grad_func=self.thederfunc, start_point=thesample.flatten())
            elif method=='adam':
                res=adamoptimize(func=self.theoptfunc,grad_func=self.thederfunc, start_point=thesample.flatten())
            else:
                res=optimize.minimize(fun=self.theoptfunc, x0=thesample.flatten(), method=method,jac=self.thederfunc,bounds=thebounds)
                exit

            print(bcolors.WARNING +"res.fun="+ bcolors.ENDC,res.fun,self.spfunc(res.x))#,giveEFels(nofmodes=nofmodes,nofAs=nofAs,Asphi=res.x,symetric=symetricG))
            if res.fun > preres: 
                if order==self.addmode and res.fun < 1.1*preres:
                    modify=True
                elif order==self.switchrandomly:
                    modify=False
                else:
                    self.plotEF(thedata=res.x)
                    print(bcolors.OKCYAN +"res.fun>preres="+ bcolors.ENDC,res.fun,self.givegatematof(Asphi=res.x))
                    print('replace and continue?')
                    answer=int(sys.stdin.readline())
                    if answer==1:
                        print('modify')
                        modify=True
                    elif answer==2:
                        print('no modify no ask')   
                        modify=False
                        askorder=False
                    else:
                        print('no modify')   
                        modify=False
                        askorder=True
            else:
                modify=True
            if res.fun<preres or modify==True:
                thesample=res.x
                presample=np.copy(res.x)
                prenofAs=self.nofAs
                prenofmodes=self.nofmodes
                preres= res.fun
                print(bcolors.OKBLUE+'modified'+ bcolors.ENDC,self.theoptfunc(res.x))
                thesample,thebounds=self.extendphiA(sample=thesample,nofaddedmode=0,nofaddedAs=0)
            else:
                thesample=np.copy(presample)
                self.nofAs=prenofAs
                self.nofmodes=prenofmodes
            if nofrandsind==nofrands and askorder:
                print(bcolors.OKBLUE +"EFels="+ bcolors.ENDC,self.givegatematof(Asphi=thesample))
                print('diff=',self.difffunc(thesample)  ,'Fid=',self.Ffunc(thesample),'sucp=',self.spfunc(thesample))
                self.plotEF(thedata=thesample)

                
                print(f'stop:{self.stop}, Add mode:{self.addmode}, Add As:{self.addAs}, switch randomly:{self.switchrandomly}, modify parameters:{self.modifying}, print the current sample:{self.printthesample}; \n insert the corresponding number')
                order=int( sys.stdin.readline())
                if order==self.switchrandomly:
                    nofrandsind=0
            if (order==self.switchrandomly):
                nofrandsind+=1
                print("to stop iteration return 1")
                i, o, e = select.select([sys.stdin], [], [], .01)
                if (i):
                    input=sys.stdin.readline().strip()
                    print("thebounds are adjusted", input)
                    if input=='1':
                        nofrandsind=nofrands
                thesample=self.switchsamplerand(nofrandswitches=nofrandswitches,sample=thesample)
            if order==self.addmode:
                nofaddedmode=nofmodestoadd
                nofaddedAs=0
                print('adding modes...')
                thesample,thebounds=self.extendphiA(sample=thesample,nofaddedmode=nofaddedmode,nofaddedAs=nofaddedAs)
                
            elif (order==self.addAs):
                nofaddedmode=0
                nofaddedAs=1
                print('adding As...')
                thesample,thebounds=self.extendphiA(sample=thesample,nofaddedmode=nofaddedmode,nofaddedAs=nofaddedAs)
            elif (order==self.printthesample):
                self.currentsample=thesample
                self.printcurrentsample()
            elif (order==self.modifying):
                print(f'qpenaltydratio={self.qpenaltydratio}, give q penalty distance ration: 0 for no change!')
                ord=float(sys.stdin.readline())
                if ord!=0:
                    self.qpenaltydratio=ord
                    preres= self.theoptfunc(presample)
                
                print(f'qpenalfac={self.qpenalfac}, give q penalty factor: 0 for no change!')
                ord=float(sys.stdin.readline())
                if ord!=0:
                    self.qpenalfac=ord
                    preres= self.theoptfunc(presample)
                print(f'spfac={self.spfac}, give success probability factor: 0 for no change!')
                ord=float(sys.stdin.readline())
                if ord!=0:
                    self.spfac=ord
                    preres= self.theoptfunc(presample)
                print(f'method={method}, give methods: 0 for no change!, 1=SLSQP,2=nelder-mead, 3=BFGS')
                print('globals: 4=dual_annealing, 5=differential_evolution')
                ord=int(sys.stdin.readline())
                if ord!=0:
                    if ord==1:
                        method='SLSQP'
                    elif ord==2:
                        method='nelder-mead'
                    elif ord==3:
                        method='BFGS'
                    elif ord==4:
                        method='da'
                    elif ord==5:
                        method='de'
                print(f'nofrands={nofrands}, give number of rands: 0 for no change!')
                ord=int(sys.stdin.readline())
                if ord!=0:
                    nofrands=ord
                print(f'nofrandswitches={nofrandswitches}, give number of random switches: 0 for no change!')
                ord=int(sys.stdin.readline())
                if ord!=0:
                    nofrandswitches=ord
                print(f'nofmodestoadd={nofmodestoadd}, give number of modes for adding: 0 for no change!')
                ord=int(sys.stdin.readline())
                if ord!=0:
                    nofmodestoadd=ord
                    
                    
                print(f'optfunc0={self.optfunc}, give optimization function: 0 for no change!, 1=F,2=diff')
                print('diff=',self.difffunc(thesample)  ,'Fid=',self.Ffunc(thesample),'sucp=',self.spfunc(thesample))
                ord=int(sys.stdin.readline())
                if ord!=0:
                    if ord==1:
                        optfunc='F'
                    elif ord==2:
                        optfunc='diff'
                    self.setoptfunction(optfunc)
                    preres= self.theoptfunc(presample)


                order=self.switchrandomly
                nofrandsind=nofrands
        self.currentsample=thesample
    def printcurrentsample(self):
        if self.thekind=='Asphi':
            print(bcolors.OKBLUE +"EFels="+ bcolors.ENDC,self.givegatematof(Asphi=self.currentsample))
            q1,q2,phi1,phi2,phi= self.getqsphisfromAsphi(Asphi=self.currentsample)
            print('q1size=',q1.size)
            print('diff=',self.difffunc(self.currentsample)  )
            print('Fid=',self.Ffunc(self.currentsample))
            print('sucp=',self.spfunc(self.currentsample))

            print('q1=',repr(q1))
            print('q2=',repr(q2))
            print('phi1=',repr(phi1))
            print('phi2=',repr(phi2))
            print('phi=',repr(phi))
            As1=self.currentsample[0:self.nofAs]
            print('As1=',repr(As1))
            if not self.symetricG:
                As2= self.currentsample[self.nofAs:2*self.nofAs]
            elif self.invG:
                As2=self.Asinv(As1)
            else:
                As2=As1
            print('As2=',repr(As2))