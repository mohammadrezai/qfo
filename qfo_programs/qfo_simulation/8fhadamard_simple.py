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

import matplotlib.pyplot as plt
import numpy as np
from source import *
from qstate import *
from qfosys import *
import scipy.special as sp
import sys
needtobechecked=False
np.set_printoptions(suppress=True)

um=0.0001
nm=0.001*um
cm=nm*10**7
mylambda=650*nm
myK=2*np.pi/mylambda
lx=100*um

#make lens
mylens= Lens(f=1*cm)
#make free
free=SLM(amps=np.array([1]),ks=np.array([0.0]))

def makegrating_Hadamard(minn=-4,maxn=4,amps=[],as1=-0.81451757):
    mydk = myK*lx/mylens.f
    if len(amps)==0:
        As1= np.array([as1])
        a1=-As1[0]  #becasuse in optimizatoin program: phifunc = - Besselfunc 
        gratingamps1=np.array([sp.jv(n, a1) for n in range(minn,maxn+1)])
    else:
        gratingamps1=np.array([amps[n] for n in range(minn,maxn+1)])
        print('sum of slm amps =',sum(abs(gratingamps1)**2))

    gratingks = np.array([n*mydk for n in range(minn,maxn+1)])
    slmgrating1=SLM(amps=gratingamps1,ks=gratingks)
    return slmgrating1

def makeslmpix(minnslm=-8,maxnslm=8,mid=0,minval=np.pi,maxval=0):    
    slmpixhadamardphis1=np.array([minval for n in range(minnslm,mid) ]) #= phi(-r) at optimizatoin program: 
    slmpixhadamardphis2=np.array([maxval for n in range(mid,maxnslm+1) ]) #= phi(-r) at optimizatoin program: 
    slmpixhadamardphis=np.concatenate((slmpixhadamardphis1,slmpixhadamardphis2), axis=None)
    pixs=np.array([n*lx for n in range(minnslm,maxnslm+1) ])
    slmpixhadamard=SLMpix(phis=slmpixhadamardphis,pixs=pixs)
    return slmpixhadamard
    

def makeslmpix_Hadamardarray(minnslm=-4,maxnslm=3,phivals=[],mid=0,minval=np.pi,maxval=0):
    if len(phivals)==0:
        slmpixhadamardphis1=np.array([minval for n in range(minnslm,mid) ])
        slmpixhadamardphis2=np.array([maxval for n in range(mid,maxnslm+1) ])
        slmpixhadamardphis=np.concatenate((slmpixhadamardphis1,slmpixhadamardphis2), axis=None)
    else:
        slmpixhadamardphis=np.array([phivals[n] for n in range(minnslm,maxnslm+1)])

    slmpixhadamardphisarray=np.concatenate((slmpixhadamardphis,slmpixhadamardphis,slmpixhadamardphis), axis=None)
    length=maxnslm-minnslm+1
    pixsarray=np.array([n*lx for n in range(minnslm-length,maxnslm+1+length) ])
    slmpixhadamard=SLMpix(phis=slmpixhadamardphisarray,pixs=pixsarray)
    #print('harrayinfo=',pixsarray.size,slmpixhadamardphisarray.size)
    return slmpixhadamard

    
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

if True: # version 2
    minn=-3
    maxn=4
    as_s= 0.81450656 #this value is calculated via optimization techniques (see program qfo_gateoptimization/opthadamard_simple.py)
    slmgrating1=makegrating_Hadamard(minn=minn,maxn=maxn,as1=as_s) 
    slmgrating2=makegrating_Hadamard(minn=minn,maxn=maxn,as1=as_s)
    slmpixhadamard=makeslmpix_Hadamardarray(minnslm=1*minn,maxnslm=1*maxn,mid=1,minval=0,maxval=-np.pi)


inputqubit=makeBasicsource(amps=[1/np.sqrt(2),-1/np.sqrt(2)],x0s=[0*lx,-1*lx])
inputqubit=makeBasicsource(amps=[1],x0s=[0*lx])
inputsource=FSource(bsources=[inputqubit],dx=lx,dy=1)
hadamardsys_1qubit_v1=Efsys(insource=inputqubit,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f)
hadamardsys_1qubit_v2=Efsys(insource=inputsource,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f)
inz=-4*mylens.f
midz=0*mylens.f
outz=4*mylens.f

xs=np.array([-0.06,0.05])


ind_exit=0
ind_showfockrepresentation=1
ind_showintensities=2
ind_showpropagation=3
ind_showslms=4
ind_saveopenvdppropagation=5
ind_showwps=6

vdbfilename='hadamardsimple.vdb'
print('To perform the desired task, please enter the corresponding number')
print(f' Exit:{ind_exit} \n Display Fock representation:{ind_showfockrepresentation} \n Display input/output intensities:{ind_showintensities} \n Display propagation :{ind_showpropagation} \n Display SLMs phase modulations:{ind_showslms} \n Save propagation as" {vdbfilename} for blender:{ind_saveopenvdppropagation} \n ')
myind=int( sys.stdin.readline())
if myind==ind_showfockrepresentation:  #for fockrepresentation and amps
    rounddig=1
    print('input Fock representation:')
    normin,totnormin=hadamardsys_1qubit_v2.printfockrepatz(z=inz,rounddig=rounddig)

    print('output Fock representation:')
    normout,totnormout=hadamardsys_1qubit_v2.printfockrepatz(z=outz,rounddig=rounddig,showy=False,absamps=False)

    print('norms=',normin,normout)
if myind==ind_showintensities: # wpx and x-intensities
    hadamardsys_1qubit_v2.plotintensityxatz(z=inz,xs=xs)
    hadamardsys_1qubit_v2.plotintensityxatz(z=outz,xs=xs)
    #hadamardsys_3qubits.plotintensityxatz(z=midz,xs=xs)

if myind==ind_showwps: # wpx and x-intensities
    hadamardsys_1qubit_v1.plotxwpsatz(z=inz,nofxbins=300,xs=xs)
    hadamardsys_1qubit_v1.plotxwpsatz(z=outz,nofxbins=300,xs=xs)
    #hadamardsys_1qubit_v1.plotxwpsatz(z=midz,nofxbins=300,xs=xs)
if myind==ind_showpropagation: # show the propagation
    hadamardsys_1qubit_v2.plotintensityx(nofzbins=100,zs=np.array([inz,outz]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),cuttoview=True)  

    if needtobechecked:
        hadamardsys_1qubit_v1=Efsys(insource=inputqubit,lensa1=mylens,slma=free,lensa2=mylens,lensb1=mylens,slmb=free,lensb2=mylens,slm=free,slmpos=0*mylens.f)
        hadamardsys_1qubit_v1.plotxintensities(zs=np.array([inz,outz]),cuttoview=True,nofzbins=100,nofxbins=300)
        hadamardsys_1qubit_v1.plotxwps(zs=np.array([inz,outz]),cuttoview=True,nofzbins=300,nofxbins=800)
        hadamardsys_1qubit_v1.plotywps(zs=np.array([inz,outz]),cuttoview=True,nofzbins=300,nofybins=800)
    
if myind==ind_showslms: #for   slm
    if True:
        xmin=slmpixhadamard.pixsx[0]
        xmax=slmpixhadamard.pixsx[-1]
        xs=np.array([xmin,xmax])
        dxxx=lx
        yticks=np.linspace(xs[0],xs[1], int((xs[1]-xs[0])/dxxx+1))
        mywsize=21.2
    else: # plotting one period
        slmperiod_k=myK*lx/mylens.f 
        slmperiod_x=2*np.pi/slmperiod_k
        xs=np.array([0,slmperiod_x])
        yticks=np.linspace(0,slmperiod_x, 11)
        mywsize=26
    xs1 ,amps1 =slmgrating1.plotwpx(xs=xs,nofbins=1000,onlygivedata=True)
    xs2 ,amps2=slmpixhadamard.plotwpx(xs=xs,onlygivedata=True)

    fig = plt.figure() 
    fig.set_figwidth(mywsize)
    fig.set_figheight(5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs1, -np.angle(amps1))
    ax.step(xs2 ,-np.angle(amps2),linestyle='-', where='mid')
    plt.grid(color = 'green', linestyle =  '-', linewidth = 0.5)
    plt.yticks([-np.pi,-3*np.pi/4,-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    plt.xticks(yticks)
    if False:  #saveslm file 
        fig.savefig(f'hadamardslms_01.svg', format='svg', dpi=1200)

if myind==ind_saveopenvdppropagation:  # save propagation  
    zs=zlim=np.array([inz,outz])
    xs=np.array([-0.08,0.08])
    gridd = vdb.FloatGrid()
    mylambdanm=int(mylambda/nm)
    print('mylambdanm=',mylambdanm)
    kind='I'
    zs,xs=hadamardsys_1qubit_v2.savedensitieswpIx(gridd=gridd,griddname=f'density_{mylambdanm}',zs=zs,xs=xs,nofzbins=800,nofxbins=1600,nofxsubbins=1,kind=kind)
    voxx=(np.max(xs)-np.min(xs))/xs.shape[0]
    voxz=(np.max(zs)-np.min(zs))/zs.shape[0]
    dx=-voxx*xs.shape[0]/2.
    dz=-voxz*zs.shape[0]/2.
    print(f'infilename=\'{vdbfilename}\';voxx= np.array([{voxx}]);voxz=np.array([{voxz}]);dx=np.array([{dx}]);dz=np.array([{dz}])')
    vdb.write(vdbfilename, gridd)
plt.show()