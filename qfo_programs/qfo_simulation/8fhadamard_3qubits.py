# Quantum Fourier Optics 

# The program calculates, from first principles, the evolution of the quantum state of light through Fourier optical systems and calculates and applies various schemes such as projection measurement and intensity measurement.
# Many equations and concepts are explained in the following papers 
# "M. Rezai and J. A. Salehi, Fundamentals of quantum Fourier Optics, IEEE Transactions on Quantum Engineering 4, 1 (2023)."
# "M. Rezai and J. A. Salehi, Quantum Computation via Multiport Quantum Fourier Optical Processors, arXiv preprint arXiv:2303.03877 (2023)."

# Copyright (C) 2023, Mohammad Rezai
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

########################### 
#########   Hadamard Gate Implementation via an 8f-Processor on Three Qubits 
#########   Figure 3 in the paper "Quantum Computation via Multiport Quantum Fourier Optical Processors"
 

from xml.dom.expatbuilder import parseString
import matplotlib.pyplot as plt

import numpy as np
from qfosys import *

import sys
from matplotlib.backends.backend_pdf import PdfPages

np.set_printoptions(suppress=True)

um=0.0001
nm=0.001*um
cm=nm*10**7
mylambda=650*nm
myK=2*np.pi/mylambda
lx=100*um

#make lens
mylens= Lens(f=2.5*cm)

def makegrating(minn,maxn,amps):
    mydk = myK*lx/mylens.f
    gratingamps1=np.array([amps[n] for n in range(minn,maxn+1)])
    gratingks = np.array([n*mydk for n in range(minn,maxn+1)])
    slmgrating=SLM(amps=gratingamps1,ks=gratingks)
    return slmgrating

def makeslmpix_array(minnslm,maxnslm,phivals):
    slmpixphis=np.array([phivals[n] for n in range(minnslm,maxnslm+1)])

    slmpixphisarray=np.concatenate((slmpixphis,slmpixphis,slmpixphis), axis=None)
    length=maxnslm-minnslm+1
    pixsarray=np.array([n*lx for n in range(minnslm-length,maxnslm+1+length) ])
    slmpix=SLMpix(phis=slmpixphisarray,pixs=pixsarray)
    return slmpix

zmin=-4*mylens.f
def makeBasicsource(amps,x0s,minwaist=40*um,z0=zmin):
    myamps=np.array(amps) 
    myx0s=np.array(x0s)
    nofss=myamps.size
    z0s=np.full((nofss), z0)
    y0s=np.full((nofss), 0.0)
    nofsigranges=np.full((nofss), 5/2)
    kx0stheta=np.full((nofss), 0.0)
    ky0stheta=np.full((nofss), 0.0)
    mysource=BSource(nofss=nofss,K=myK,z0s=z0s,x0s=myx0s,y0s=y0s,kx0stheta=kx0stheta,ky0stheta=ky0stheta,nofsigranges=nofsigranges,amps=myamps,longphase=False)
    mysource.setminwaistsx(minwaists= np.full((nofss), minwaist),dztominwaists=np.full((nofss), 0.0))
    return mysource
def gratingfromAs(phifunctype,As,xs):
    mydk = myK*lx/mylens.f
    k0x=mydk*xs
    phi=np.zeros_like(k0x,dtype=np.float64)
    if phifunctype=='sin':
        for i in range(As.size):
            phi+=As[i]*np.sin((i+1)*k0x)
    elif phifunctype=='sincos':
        for i in range(As.size):
            if i%2!=0:
                phi+=As[i]*np.cos((int(i/2)+1)*k0x)
            else:
                phi+=As[i]* np.sin((int(i/2)+1)*k0x)

    elif phifunctype=='cos':
        for i in range(As.size):
            phi+=As[i]* np.sin((i+1)*k0x)
    else:
        ex=1/0
    return phi
        


#make basicsources qubits
myshift=1
qubit0=makeBasicsource(amps=[1],x0s=[(0+myshift)*lx])
qubitup=makeBasicsource(amps=[1/np.sqrt(2),1/np.sqrt(2)],x0s=[(2+myshift)*lx,(1+myshift)*lx])
qubitdown=makeBasicsource(amps=[1/np.sqrt(2),-1/np.sqrt(2)],x0s=[(-2+myshift)*lx,(-3+myshift)*lx])
# make a 3qubit factorized source for the input
inputsource=FSource(bsources=[qubit0,qubitup,qubitdown],dx=lx,dy=1,bsourcesold=[],activeold=False)

# Making Hadamard 8fsys. 
# The following q1, As1 and phi values are evaluated via optimization techniques (see program qfo_gateoptimization/opthadamard_3qubits.py)
maxn=12
minn=-13
q1= np.array([-0.13046362+0.06930189j,  0.06417221+0.16985863j,
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
As1= np.array([ 0.74686637,  0.02568929,  0.11389472, -2.74314672, -0.07029734,
        0.00149469, -0.08427034, -0.0031854 , -0.00839394, -0.00489992,
        0.00052054])
samplephifunctype='sincos'

slmgrating1=makegrating(minn=minn,maxn=maxn,amps=q1)
slmgrating2=makegrating(minn=minn,maxn=maxn,amps=q1)
phi= np.array([ 3.09875438, -0.06319974,  0.21503727, -3.31023343, -2.92965553,
    -0.01306055,  0.28801485, -3.03851175, -2.62412132,  0.12453885,
        0.78753411, -3.03199734, -2.09130288,  0.07542848,  1.28456214,
    -3.10837991, -1.66398162, -0.00522035,  1.49588591, -2.95818386,
    -0.90576413,  2.64895801, -2.62171063, -0.28771298,  1.74417051,
        1.98973479, -0.27433307, -1.20595896,  2.93200871,  2.07274102,
    -0.16441243, -0.93206777,  3.00628887,  2.34448289, -0.12601783,
    -0.66454093,  3.00298854,  2.6040078 , -0.1677251 , -0.4303796 ,
        2.96065436,  2.74429811, -0.06524269, -0.3701864 ,  3.22372463])
phi=np.roll(phi,-1*myshift)
gaterange=5
slmpixhadamard=makeslmpix_array(minnslm=1*(minn-gaterange),maxnslm=1*(maxn+gaterange),phivals=phi)

# making The Hadamard system with 3 qubits input
hadamardsys_3qubits=Efsys(insource=inputsource,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f)

inz=-4*mylens.f
midz=0*mylens.f
outz=4*mylens.f
xs=np.array([-0.05,0.05])

ind_exit=0
ind_showfockrepresentation=1
ind_showintensities=2
ind_showpropagation=3
ind_showslms=4
ind_saveopenvdppropagation=5

vdbfilename='hadamard_3qubits.vdb'
print('To perform the desired task, please enter the corresponding number')
print(f' Exit:{ind_exit} \n Display Fock representation:{ind_showfockrepresentation} \n Display input/output intensities:{ind_showintensities} \n Display propagation :{ind_showpropagation} \n Display SLMs phase modulations:{ind_showslms} \n Save propagation as" {vdbfilename} for blender:{ind_saveopenvdppropagation} \n ')
myind=int( sys.stdin.readline())

if myind==ind_showfockrepresentation: #for fockrepresentation and amps
    rounddig=1
    print('input Fock representation:')
    normin,totnormin=hadamardsys_3qubits.printfockrepatz(z=inz,rounddig=rounddig)

    print('output Fock representation:')
    normout,totnormout=hadamardsys_3qubits.printfockrepatz(z=outz,rounddig=rounddig,showy=False,absamps=False)

    print('norms=',normin,normout)
    
if myind==ind_showintensities: # wpx and x-intensities
    hadamardsys_3qubits.plotintensityxatz(z=inz,xs=xs)
    hadamardsys_3qubits.plotintensityxatz(z=outz,xs=xs)
    #hadamardsys_3qubits.plotintensityxatz(z=midz,xs=xs)

if myind==ind_showpropagation: # show the propagation
    hadamardsys_3qubits.plotintensityx(nofzbins=100,zs=np.array([inz,outz]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),cuttoview=True)

if myind==ind_showslms: #for   slm
    if True:
        xs=np.array([-0.17,0.17])
        dxxx=lx
        yticks=np.linspace(xs[0],xs[1], int((xs[1]-xs[0])/dxxx+1))
        mywsize=21.2
    else: # plotting one period
        slmperiod_k=myK*lx/mylens.f 
        slmperiod_x=2*np.pi/slmperiod_k
        xs=np.array([0,slmperiod_x])
        yticks=np.linspace(0,slmperiod_x, 11)
        mywsize=26
    def phisslmgrating1(xs):
        return gratingfromAs(phifunctype=samplephifunctype,As=As1,xs=xs)
    xs1 ,amps1 =slmgrating1.plotwpx(xs=xs,nofbins=1000,onlygivedata=True)
    phiss1=phisslmgrating1(xs=xs1)
    xs2 ,amps2=slmpixhadamard.plotwpx(xs=xs,onlygivedata=True)

    fig = plt.figure() 
    fig.set_figwidth(mywsize)
    fig.set_figheight(5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs1, phiss1)
    ax.step(xs2 ,-np.angle(amps2),linestyle='-', where='mid')
    plt.grid(color = 'green', linestyle =  '-', linewidth = 0.5)
    plt.yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    plt.xticks(yticks)
    if False:  #saveslm file 
        fig.savefig(f'hadamardslms_02.svg', format='svg', dpi=1200)

if myind==ind_saveopenvdppropagation:  # save propagation  
    nofxbs=3000   #this value is used for fig. 4; to change the number of voxels in x direction, change this value
    nofzbs=800 #this value is used for fig. 4; to change the number of voxels in z direction, change this value
    answer=1
    if nofxbs*nofzbs>=4000:
        print (" Warning:\n The number of voxels is big; Therefore, saving data in the vdb format probably takes several hours. \n To resume return 1, otherwise, return 0 to stop the program. \n To reduce the execution time decrease nofxbs and/or nofzbs in the program.")
        answer=int( sys.stdin.readline())
    if answer==1:
        zs=zlim=np.array([inz,outz])
        xs=np.array([-0.17,0.17])
        gridd = vdb.FloatGrid()
        mylambdanm=int(mylambda/nm)
        print('mylambdanm=',mylambdanm)
        kind='I'
        zs,xs=hadamardsys_3qubits.savedensitieswpIx(gridd=gridd,griddname=f'density_{mylambdanm}',zs=zs,xs=xs,nofzbins=nofzbs,nofxbins=nofxbs,nofxsubbins=1,kind=kind)
        
        voxx=(np.max(xs)-np.min(xs))/xs.shape[0]
        voxz=(np.max(zs)-np.min(zs))/zs.shape[0]
        dx=-voxx*xs.shape[0]/2.
        dz=-voxz*zs.shape[0]/2.
        print(f'infilename=\'{vdbfilename}\';voxx= np.array([{voxx}]);voxz=np.array([{voxz}]);dx=np.array([{dx}]);dz=np.array([{dz}])')
        vdb.write(vdbfilename, gridd)
plt.show()