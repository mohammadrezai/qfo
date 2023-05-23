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
from numpy import array
from scipy.fft import fft, ifft
import scipy.special as sp
import sys

#from sympy import false

np.set_printoptions(suppress=True)

Hm=np.matrix([
    [1,1],
    [1,-1]],dtype=np.float64)/(2**(1/2))
iisH=[0,-1]
jjsH=[0,-1]
Gshifts=[0]


from qfogateoptimization import GateArray
from qfogateoptimization import optimizer

optfunc='diff'

ga=GateArray(Gates=[Hm],gatetype='efarr',Giis=[iisH],Gjjs=[jjsH],Gshifts=Gshifts,ijnormelG=[(0,0)])


print('To take the optimal value presented in the paper as the initial guess for the optimization, return 1 otherwise, return 0.')
answer=int(sys.stdin.readline())
if answer==1:
    usecurrentsample=True
    As1= np.array([0.81450656])
    nofmodes=8
    phi=np.empty(nofmodes,dtype=float)
    phi[0]=0
    nofmodes2=int(nofmodes/2)+1
    for i in range(nofmodes):
        if i==0:
            phi[i]=0
        elif i<nofmodes2:
            phi[i]=-np.pi
        else:
            phi[i]=0    
    
    theoptimizer=optimizer(GA=ga,nofAs0=As1.size,nofmodes0=phi.size,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')
    theoptimizer.setcurrentsample(symetricG=True,phi=phi,As1=As1,As2=As1,samplephifunctype='sincos')
    theoptimizer.printcurrentsample()
    theoptimizer.plotEF(theoptimizer.currentsample)
else:
    usecurrentsample=False
    theoptimizer=optimizer(GA=ga,nofAs0=1,nofmodes0=8,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')

if True:
    method='adam'
    nofrands=1
    nofrandswitches=3
    theoptimizer.opt(method=method,nofmodestoadd=1,usecurrentsample=usecurrentsample,nofrands=nofrands,nofrandswitches=nofrandswitches)
    theoptimizer.printcurrentsample()