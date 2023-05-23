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
#########   CNOT Gate Implementation via an 8f-Processor and a Projective Measurement 
#########   Figure 4 in the paper "Quantum Computation via Multiport Quantum Fourier Optical Processors"
 
import matplotlib.pyplot as plt
import numpy as np
from source import *
from qstate import *
from qfosys import *
import sys
np.set_printoptions(suppress=True)

um=0.0001
nm=0.001*um
cm=nm*10**7
mylambda=650*nm
myK=2*np.pi/mylambda
lx=100*um

#make lens
mylens= Lens(f=2*cm)

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

#make basicsources
controlqubit=makeBasicsource(amps=[1/np.sqrt(2),1/np.sqrt(2)],x0s=[0*lx,1*lx])
targetqubit=makeBasicsource(amps=[1],x0s=[-1*lx])

inputsource_v1=FSource(bsources=[controlqubit,targetqubit],dx=lx,dy=1)

# Making CNOT 8fsys. 
# The following q1, As1 and phi values are evaluated via optimization techniques (see program qfo_gateoptimization/optcnot.py)

if True:  # make sys
    maxn=10
    minn=-10
    gaterange=2
    q1= np.array([ 0.62369165-0.0477945j ,  0.46153801+0.1048348j ,
        0.0817916 -0.20049788j,  0.13674067-0.12267962j,
        0.01617611-0.0193315j , -0.01920167-0.03606649j,
        0.00033655-0.00853501j, -0.00503315+0.0026464j ,
       -0.00255044-0.00075144j,  0.0001718 +0.00049974j,
       -0.00018   +0.00048039j,  0.00000005+0.00000003j,
       -0.        +0.00000002j,  0.00011891+0.00001313j,
       -0.00012768-0.00031314j,  0.00077437+0.00125802j,
       -0.00149769-0.0015699j , -0.00096167+0.00515147j,
        0.00000501-0.02194119j,  0.00152304+0.0303163j ,
        0.05054712-0.03333327j, -0.10133383+0.14814095j,
        0.07394689-0.20440256j, -0.45442103-0.10294012j])
    slmgrating1=makegrating(minn=minn,maxn=maxn,amps=q1)
    slmgrating2=makegrating(minn=minn,maxn=maxn,amps=q1)

    phi= np.array([-2.55274722, -4.38923871, -4.3098047 ,  3.88545399, -2.28621385,
        7.20697282,  5.88039845,  4.64971185,  1.62427446,  1.3693039 ,
        4.4416976 , -0.19621886, -1.55951573,  1.18101934,  5.30157507,
       -1.25831835,  5.31701086,  0.36409614,  0.26921557,  1.95310151,
        2.24841016,  1.60494364, -4.55908215, -2.50788026])
    slmpixhadamard=makeslmpix_array(minnslm=1*(minn-gaterange),maxnslm=1*(maxn+gaterange),phivals=phi)
    
    if True:  # make hadamard 8f-system
        #mysysFFock=Efsys(insource=inputsource_v1,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f)
        #mysysEFock=Efsys(insource=inputsource_v2,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f)
        projns=[[1,1],[1,1],[1,1],[1,1]]
        projxs=[[1,-1],[1,-2],[0,-1],[0,-2]]
        projys=[[0,0],[0,0],[0,0],[0,0]]

        cnotsys=EfEProjsys(insource=inputsource_v1,lensa1=mylens,slma=slmgrating1,lensa2=mylens,lensb1=mylens,slmb=slmgrating2,lensb2=mylens,slm=slmpixhadamard,slmpos=0*mylens.f,projns=projns,projxs=projxs,projys=projys,demolish=False)

inz=-4*mylens.f
midz=0*mylens.f
outz=4*mylens.f
xs=np.array([-0.1,0.1])

ind_exit=0
ind_showfockrepresentation=1
ind_showintensities=2
ind_showpropagation=3
ind_showslms=4
ind_saveopenvdppropagation=5

vdbfilename='cnot.vdb'
print('To perform the desired task, please enter the corresponding number')
print(f' Exit:{ind_exit} \n Display Fock representation:{ind_showfockrepresentation} \n Display input/output intensities:{ind_showintensities} \n Display propagation :{ind_showpropagation} \n Display SLMs phase modulations:{ind_showslms} \n Save propagation as" {vdbfilename} for blender:{ind_saveopenvdppropagation} \n ')
myind=int( sys.stdin.readline())

if myind==ind_showfockrepresentation: #for fockrepresentation and amps
    rounddig=2
    print('input Fock representation:')
    normin,totnormin=cnotsys.printfockrepatz(z=inz,rounddig=rounddig)

    print('output Fock representation:')
    normout,totnormout=cnotsys.printfockrepatz(z=outz,rounddig=rounddig,showy=False,absamps=False)
    #        normax=cnotsys.printfockrepatz(z=outz,rounddig=rounddig,absamps=True)
    print('norms=',normin,normout)
if myind==ind_showintensities: # wpx and x-intensities
    cnotsys.plotintensityxatz(z=inz,xs=xs)
    cnotsys.plotintensityxatz(z=outz,xs=xs)
    #cnotsys.plotintensityxatz(z=midz,xs=xs)
if myind==ind_showpropagation: # show the propagation
    cnotsys.plotintensityx(nofzbins=100,zs=np.array([inz,outz+mylens.f/4]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),cuttoview=True)
if myind==ind_showslms: #for   slm
    if True:
        xs=np.array([-0.1,0.1])
        dxxx=lx
        yticks=np.linspace(xs[0],xs[1], int((xs[1]-xs[0])/dxxx+1))
        mywsize=17.8
    else: # plotting one period
        slmperiod_k=myK*lx/mylens.f 
        slmperiod_x=2*np.pi/slmperiod_k
        xs=np.array([0,slmperiod_x])
        yticks=np.linspace(0,slmperiod_x, 11)
        mywsize=17.8
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
        fig.savefig(f'cnothslms_02.svg', format='svg', dpi=1200)


if myind==ind_saveopenvdppropagation:  # save propagation  
    normalize=123
    extendration=50
    nofxbs=4500   #this value is used for fig. 4; to change the number of voxels in x direction, change this value
    nofzbs=800+extendration #this value is used for fig. 4; to change the number of voxels in z direction, change this value
    answer=1
    if nofxbs*nofzbs>=4000:
        print (" 5Warning:\n The number of voxels is big; Therefore, saving data in the vdb format probably takes several hours. \n To resume return 1, otherwise, return 0 to stop the program. \n To reduce the execution time decrease nofxbs and/or nofzbs in the program.")
        answer=int( sys.stdin.readline())
    if answer==1:
        zs=zlim=np.array([zmin,outz+mylens.f*extendration/100])

        xs=np.array([-0.1,0.1])
        gridd = vdb.FloatGrid()
        mylambdanm=int(mylambda/nm)
        print('mylambdanm=',mylambdanm)
        kind='I'
        
        zs,xs=cnotsys.savedensitieswpIx(gridd=gridd,griddname=f'density_{mylambdanm}',zs=zs,xs=xs,nofzbins=nofzbs,nofxbins=nofxbs,nofxsubbins=1,kind=kind,normalize=normalize,projz=4*mylens.f)

        voxx=(np.max(xs)-np.min(xs))/xs.shape[0]
        voxz=(np.max(zs)-np.min(zs))/zs.shape[0]
        dx=-voxx*xs.shape[0]/2.
        dz=-voxz*zs.shape[0]/2.
        print(f'infilename=\'{vdbfilename}\';voxx= np.array([{voxx}]);voxz=np.array([{voxz}]);dx=np.array([{dx}]);dz=np.array([{dz}])')
        vdb.write(vdbfilename, gridd)

plt.show()