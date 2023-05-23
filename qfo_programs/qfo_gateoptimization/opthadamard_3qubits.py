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

########################### 
#########   Hadamard Gate Implementation via an 8f-Processor on Three Qubits 
#########   Figure 3 in the paper "Quantum Computation via Multiport Quantum Fourier Optical Processors"


import numpy as np 
from numpy import array
from scipy.fft import fft, ifft
import scipy.special as sp
import sys

np.set_printoptions(suppress=True)

Hm=np.matrix([
    [1,1],
    [1,-1]],dtype=np.float64)/(2**(1/2))
iisH=[0,-1]
jjsH=[0,-1]
Gshifts=[2,0,-2]
myshift=1

from qfogateoptimization import GateArray
from qfogateoptimization import optimizer

optfunc='diff'
for ish in range(len(Gshifts)):
    Gshifts[ish]+=myshift
print('newGshifts',Gshifts)

ga=GateArray(Gates=[Hm,Hm,Hm],gatetype='efarr',Giis=[iisH,iisH,iisH],Gjjs=[jjsH,jjsH,jjsH],Gshifts=Gshifts,ijnormelG=[(0,0),(0,0),(0,0)])


print('To take the optimal value presented in the paper as the initial guess for the optimization, return 1 otherwise, return 0.')
answer=int(sys.stdin.readline())
if answer==1:
    usecurrentsample=True
    q1size= 45
    diff= 0.0003630102077217417
    Fid= 0.9999892137424484
    sucp= 0.9900027851070824
    q1= array([-0.13046362+0.06930189j,  0.06417221+0.16985863j,
        -0.06717124+0.39058878j, -0.16118649-0.15673015j,
        -0.41779145-0.01906013j,  0.17135785-0.05515056j,
        -0.00428561-0.21878587j,  0.00181217+0.0894305j ,
            0.07671776-0.00581097j, -0.03084472-0.00614891j,
            0.00263184+0.02017926j,  0.00304827-0.00786685j,
        -0.00423826+0.00077029j,  0.00157463+0.00087041j,
        -0.00016479-0.00074119j, -0.00017675+0.00025744j,
            0.00011122-0.00002696j, -0.00003553-0.00002727j,
            0.00000348+0.00001466j,  0.00000342-0.0000042j ,
        -0.00000138+0.00000019j,  0.00000094-0.00000041j,
        -0.00000092-0.00000211j, -0.0000041 -0.00000315j,
        -0.0000109 +0.00000454j, -0.00001944+0.00002109j,
            0.00002171+0.00005678j,  0.00010013+0.00010833j,
            0.00026988-0.00009646j,  0.00054863-0.0004311j ,
        -0.00039796-0.00115883j, -0.00165623-0.00249823j,
        -0.00442489+0.00151888j, -0.01007473+0.00554888j,
            0.00532173+0.01469951j,  0.01565942+0.03520368j,
            0.04112004-0.01682993j,  0.10311245-0.03512399j,
        -0.04645008-0.09176573j, -0.05564142-0.23965322j,
        -0.14664051+0.10510216j, -0.39696039+0.04234431j,
            0.17250563+0.12045306j, -0.03307969+0.34300724j,
        -0.06245971-0.14590708j])
    q2= array([-0.13046362+0.06930189j,  0.06417221+0.16985863j,
        -0.06717124+0.39058878j, -0.16118649-0.15673015j,
        -0.41779145-0.01906013j,  0.17135785-0.05515056j,
        -0.00428561-0.21878587j,  0.00181217+0.0894305j ,
            0.07671776-0.00581097j, -0.03084472-0.00614891j,
            0.00263184+0.02017926j,  0.00304827-0.00786685j,
        -0.00423826+0.00077029j,  0.00157463+0.00087041j,
        -0.00016479-0.00074119j, -0.00017675+0.00025744j,
            0.00011122-0.00002696j, -0.00003553-0.00002727j,
            0.00000348+0.00001466j,  0.00000342-0.0000042j ,
        -0.00000138+0.00000019j,  0.00000094-0.00000041j,
        -0.00000092-0.00000211j, -0.0000041 -0.00000315j,
        -0.0000109 +0.00000454j, -0.00001944+0.00002109j,
            0.00002171+0.00005678j,  0.00010013+0.00010833j,
            0.00026988-0.00009646j,  0.00054863-0.0004311j ,
        -0.00039796-0.00115883j, -0.00165623-0.00249823j,
        -0.00442489+0.00151888j, -0.01007473+0.00554888j,
            0.00532173+0.01469951j,  0.01565942+0.03520368j,
            0.04112004-0.01682993j,  0.10311245-0.03512399j,
        -0.04645008-0.09176573j, -0.05564142-0.23965322j,
        -0.14664051+0.10510216j, -0.39696039+0.04234431j,
            0.17250563+0.12045306j, -0.03307969+0.34300724j,
        -0.06245971-0.14590708j])
    phi1= array([-2.72404806, -2.55945236, -2.17238556, -1.57802093, -0.81547183,
            0.0551466 ,  0.95955895,  1.8187044 ,  2.55868558,  3.11931568,
            3.45981115,  3.56103811,  3.42461946,  3.06986347,  2.52971204,
            1.84672471,  1.0696639 ,  0.2507526 , -0.55666422, -1.30041938,
        -1.93203689, -2.40950989, -2.70021947, -2.78377442, -2.65450586,
        -2.32329049, -1.81827462, -1.18398182, -0.47833168,  0.23261311,
            0.88180743,  1.40926788,  1.76946314,  1.93641846,  1.90536108,
            1.69065799,  1.32087882,  0.83264689,  0.26518483, -0.3429931 ,
        -0.95477244, -1.53317079, -2.03959964, -2.4333611 , -2.67361173])
    phi2= array([-2.72404806, -2.55945236, -2.17238556, -1.57802093, -0.81547183,
            0.0551466 ,  0.95955895,  1.8187044 ,  2.55868558,  3.11931568,
            3.45981115,  3.56103811,  3.42461946,  3.06986347,  2.52971204,
            1.84672471,  1.0696639 ,  0.2507526 , -0.55666422, -1.30041938,
        -1.93203689, -2.40950989, -2.70021947, -2.78377442, -2.65450586,
        -2.32329049, -1.81827462, -1.18398182, -0.47833168,  0.23261311,
            0.88180743,  1.40926788,  1.76946314,  1.93641846,  1.90536108,
            1.69065799,  1.32087882,  0.83264689,  0.26518483, -0.3429931 ,
        -0.95477244, -1.53317079, -2.03959964, -2.4333611 , -2.67361173])
    phi= array([ 3.09875438, -0.06319974,  0.21503727, -3.31023343, -2.92965553,
        -0.01306055,  0.28801485, -3.03851175, -2.62412132,  0.12453885,
            0.78753411, -3.03199734, -2.09130288,  0.07542848,  1.28456214,
        -3.10837991, -1.66398162, -0.00522035,  1.49588591, -2.95818386,
        -0.90576413,  2.64895801, -2.62171063, -0.28771298,  1.74417051,
            1.98973479, -0.27433307, -1.20595896,  2.93200871,  2.07274102,
        -0.16441243, -0.93206777,  3.00628887,  2.34448289, -0.12601783,
        -0.66454093,  3.00298854,  2.6040078 , -0.1677251 , -0.4303796 ,
            2.96065436,  2.74429811, -0.06524269, -0.3701864 ,  3.22372463])
    As1= array([ 0.74686637,  0.02568929,  0.11389472, -2.74314672, -0.07029734,
            0.00149469, -0.08427034, -0.0031854 , -0.00839394, -0.00489992,
            0.00052054])
    As2= array([ 0.74686637,  0.02568929,  0.11389472, -2.74314672, -0.07029734,
            0.00149469, -0.08427034, -0.0031854 , -0.00839394, -0.00489992,
            0.00052054])
    phi=np.roll(phi,-1*myshift)
    
    theoptimizer=optimizer(GA=ga,nofAs0=As1.size,nofmodes0=phi.size,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')
    theoptimizer.setcurrentsample(symetricG=True,phi=phi,As1=As1,As2=As1,samplephifunctype='sincos')
    theoptimizer.printcurrentsample()
    theoptimizer.plotEF(theoptimizer.currentsample)
else:
    usecurrentsample=False
    theoptimizer=optimizer(GA=ga,nofAs0=4,nofmodes0=17,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')



if True:
    method='adam'
    nofrands=1
    nofrandswitches=3
    theoptimizer.opt(method=method,nofmodestoadd=1,usecurrentsample=usecurrentsample,nofrands=nofrands,nofrandswitches=nofrandswitches)
    theoptimizer.printcurrentsample()