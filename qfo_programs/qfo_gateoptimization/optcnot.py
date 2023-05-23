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
#########   CNOT Gate Implementation via an 8f-Processor and a Projective Measurement 
#########   Figure 4 in the paper "Quantum Computation via Multiport Quantum Fourier Optical Processors"

import numpy as np 
from numpy import array
from scipy.fft import fft, ifft
import scipy.special as sp
import sys

np.set_printoptions(suppress=True)

iisCx=[1,0,-1,-2]#[0,1,2,3]
jjsCx=[1,0,-1,-2]#[0,1,2,3]

optfunc='diff'
method='SLSQP'
method='adam'
from qfogateoptimization import GateArray
from qfogateoptimization import optimizer,qcx

Gshifts=[0]
ga=GateArray(Gates=[qcx],gatetype='cxh',Giis=[iisCx],Gjjs=[jjsCx],Gshifts=Gshifts,ijnormelG=[(0,0)])

print('To take the optimal value presented in the paper as the initial guess for the optimization, return 1 otherwise, return 0.')
answer=int(sys.stdin.readline())

if answer==1:
    usecurrentsample=True
    q1size= 24
    diff= 1.779213554011482
    Fid= -0.9997479021388161
    sucp= 0.11098423869345089
    q1= array([ 0.62369165-0.0477945j ,  0.46153801+0.1048348j ,
            0.0817916 -0.20049789j,  0.13674067-0.12267962j,
            0.01617611-0.0193315j , -0.01920167-0.03606649j,
            0.00033655-0.00853501j, -0.00503315+0.0026464j ,
        -0.00255044-0.00075144j,  0.0001718 +0.00049974j,
        -0.00018   +0.00048039j,  0.00000005+0.00000003j,
        -0.        +0.00000002j,  0.00011891+0.00001313j,
        -0.00012768-0.00031314j,  0.00077437+0.00125802j,
        -0.00149769-0.0015699j , -0.00096167+0.00515147j,
            0.00000501-0.02194119j,  0.00152304+0.0303163j ,
            0.05054711-0.03333327j, -0.10133383+0.14814095j,
            0.07394689-0.20440257j, -0.45442103-0.10294012j])
    q2= array([ 0.62369165-0.0477945j ,  0.46153801+0.1048348j ,
            0.0817916 -0.20049789j,  0.13674067-0.12267962j,
            0.01617611-0.0193315j , -0.01920167-0.03606649j,
            0.00033655-0.00853501j, -0.00503315+0.0026464j ,
        -0.00255044-0.00075144j,  0.0001718 +0.00049974j,
        -0.00018   +0.00048039j,  0.00000005+0.00000003j,
        -0.        +0.00000002j,  0.00011891+0.00001313j,
        -0.00012768-0.00031314j,  0.00077437+0.00125802j,
        -0.00149769-0.0015699j , -0.00096167+0.00515147j,
            0.00000501-0.02194119j,  0.00152304+0.0303163j ,
            0.05054711-0.03333327j, -0.10133383+0.14814095j,
            0.07394689-0.20440257j, -0.45442103-0.10294012j])
    phi1= array([ 0.53148655,  0.05416631, -0.49389733, -0.99215694, -1.32445044,
        -1.45827235, -1.45390857, -1.38389861, -1.24974976, -0.9862944 ,
        -0.54490181,  0.03138844,  0.60548949,  1.00898349,  1.12548254,
            0.95651522,  0.63757058,  0.37707147,  0.33452316,  0.51982847,
            0.79965357,  1.00434548,  1.03270201,  0.86832342])
    phi2= array([ 0.53148655,  0.05416631, -0.49389733, -0.99215694, -1.32445044,
        -1.45827235, -1.45390857, -1.38389861, -1.24974976, -0.9862944 ,
        -0.54490181,  0.03138844,  0.60548949,  1.00898349,  1.12548254,
            0.95651522,  0.63757058,  0.37707147,  0.33452316,  0.51982847,
            0.79965357,  1.00434548,  1.03270201,  0.86832342])
    phi= array([-2.55274722, -4.38923871, -4.3098047 ,  3.88545399, -2.28621385,
            7.20697282,  5.88039845,  4.64971185,  1.62427446,  1.3693039 ,
            4.4416976 , -0.19621886, -1.55951573,  1.18101934,  5.30157507,
        -1.25831835,  5.31701086,  0.36409614,  0.26921557,  1.95310151,
            2.24841016,  1.60494364, -4.55908215, -2.50788026])
    As1= array([-1.1434865 , -0.01108182, -0.0134232 ,  0.56409036, -0.23475866,
        -0.02689754,  0.05493036,  0.00439766,  0.01451197,  0.00097789])
    As2= array([-1.1434865 , -0.01108182, -0.0134232 ,  0.56409036, -0.23475866,
        -0.02689754,  0.05493036,  0.00439766,  0.01451197,  0.00097789])  
    

    theoptimizer=optimizer(GA=ga,nofAs0=As1.size,nofmodes0=phi.size,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')
    theoptimizer.setcurrentsample(symetricG=True,phi=phi,As1=As1,As2=As1,samplephifunctype='sincos')
    theoptimizer.printcurrentsample()
    theoptimizer.plotEF(theoptimizer.currentsample)
else:
    usecurrentsample=False
    theoptimizer=optimizer(GA=ga,nofAs0=4,nofmodes0=17,symetricG=True,optfunc=optfunc,Asrangemin=-.01,Asrangemax=.01,phisrangemin=-np.pi,phisrangemax=np.pi,thekind='Asphi',phifunctype='sincos')

    
if True:
    nofrands=1
    nofrandswitches=4

    theoptimizer.opt(method=method,nofmodestoadd=2,usecurrentsample=usecurrentsample,nofrands=nofrands,nofrandswitches=nofrandswitches)
    theoptimizer.printcurrentsample()