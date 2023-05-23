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
 
import sys
import copy
import numpy as np
import gbeam  
from source import *
from qstate import *
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from plot import *
useopenvdb=True
if useopenvdb: 
    sys.path.append('/usr/local/lib/python3.11/site-packages/')
    import pyopenvdb as vdb
import colorsys

cmap = plt.get_cmap('hsv')
xunit='cm'

class Lens:
    def __init__(self,f):
        self.f=f
class SLMpix:
    def __init__(self,phis,pixs):
        self.xunit=xunit
        self.size=phis.size
        self.phis=phis     
        self.amps=np.exp(-1j *phis)
        if type(phis[0])==np.float64: # 1d slm
            self.dim=1
            self.pixsx=pixs 
            self.ampx=self.amps
            self.ampy=[1]
            self.phisx=self.phis
            self.phisy=[0]
        else: # 2d slm
            self.dim=2
            self.pixsx=pixs[0,:]
            self.pixsy=pixs[1,:]
            self.ampx=self.amps[0,:]
            self.ampy=self.amps[1,:]
            self.phisx=self.phis[0,:]
            self.phisy=self.phis[1,:]
    def plotwpx(self,xs=np.array([]),onlygivedata=False,save=False,filename='slmpix'):
        if xs.size==0:
            xs=self.pixsx
            amps=self.ampx
            phis=self.phisx
        elif xs.size==2:
            minx=np.min(xs) 
            maxx=np.max(xs)
            ix=0
            dx=self.pixsx[1]-self.pixsx[0]
            while ix<self.pixsx.size and self.pixsx[ix]<minx-dx/2:
                ix+=1
            xs=np.array([],dtype=np.float64)
            amps=np.array([],dtype=np.complex128)
            phis=np.array([],dtype=np.float64)
            while ix<self.pixsx.size and self.pixsx[ix]<=maxx+5*dx/8:
                xs=np.append(xs,self.pixsx[ix])
                amps=np.append(amps,self.ampx[ix])
                phis=np.append(phis,self.phisx[ix])
                ix+=1
        else:
            err=1/0
       
        if onlygivedata:
            return xs, amps
        else:
            fig = plt.figure() 
            ax = fig.add_subplot(1, 2, 1)
            fig.suptitle('SLMPix(x)')
            phasedplotv2(ax,xs,amps,kind='x',xunit=self.xunit, minphf=-1, maxphf=1) 
            ax = fig.add_subplot(1, 2, 2)
            fig.suptitle('SLMPIX(x)')
            plotv2(ax,xs,phis,kind='phi',xunit=self.xunit,step=True)
            if save :
                plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
                fig.savefig(f'{filename}.svg', format='svg', dpi=1200)

class SLM:
    def __init__(self,amps,ks):
        self.xunit=xunit
        if type(ks[0])==np.float64: # 1d slm
            self.dim=1
            self.kx=ks
            self.ampx=amps
            self.ampy=[1]
            self.ky=[0]
            self.size=amps.size
            self.amps=amps
            self.ks=ks
            
        else: # 2d slm
            self.dim=2
            self.kx=ks[0]#[0,:]
            self.ky=ks[1]#[1,:]
            self.ampx=amps[0]#[0,:]
            self.ampy=amps[1]#[1,:]
            self.size=amps[0].size*amps[1].size
            self.amps=np.empty(self.size,dtype=complex)
            self.ks=np.empty([2,self.size],dtype=float)
            kk=-1
            for ii in range (amps[0].size):
                for jj in range(amps[1].size):
                    kk+=1
                    self.amps[kk]=amps[0][ii]*amps[1][jj]
                    self.ks[0][kk]=ks[0][ii]
                    self.ks[1][kk]=ks[1][jj]
    def wpx(self,xs):
        outamps=np.zeros((xs.size),dtype=complex)
        for ss in range(self.size):
            ii=-1
            for x in xs:
                ii+=1
                if self.ks.ndim==1:
                    outamps[ii]+=self.amps[ss]*np.exp(1j*self.ks[ss]*x)
                else: 
                    print("ERROR: 2D SLM")
                    sys.exit()
        return outamps
    def wpxy(self,xs,ys):
        outamps=np.zeros((xs.size,ys.size),dtype=complex)
        for ss in range(self.size):
            ii=-1
            for x in xs:
                ii+=1
                jj=-1
                for y in ys:
                    jj+=1
                    if self.ks.ndim==1:
                        outamps[ii,jj]+=self.amps[ss]*np.exp(1j*self.ks[ss]*x)
                    else: 
                        outamps[ii,jj]+=self.amps[ss]*np.exp(1j*self.ks[0][ss]*x)*np.exp(1j*self.ks[1][ss]*y)
        return outamps
    def wpxy(self,xs,ys):
        outamps=np.zeros((xs.size,ys.size),dtype=complex)
        for ss in range(self.size):
            ii=-1
            for x in xs:
                ii+=1
                jj=-1
                for y in ys:
                    jj+=1
                    if self.ks.ndim==1:
                        outamps[ii,jj]+=self.amps[ss]*np.exp(1j*self.ks[ss]*x)
                    else: 
                        outamps[ii,jj]+=self.amps[ss]*np.exp(1j*self.ks[0][ss]*x)*np.exp(1j*self.ks[1][ss]*y)
        return outamps
    def plotwpx(self,xs,nofbins=300,rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False,save=False,filename='slm'):
        if xs.size<3:
            minx=np.min(xs) 
            maxx=np.max(xs) 
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofbins)
           
        rescalesign= -1 if paritytransform else 1  
        amps=rescaleamp*self.wpx(xs*abs(rescaleamp)**2*rescalesign)
        if onlygivedata:
            return xs, amps
        else:
            fig = plt.figure() 
            ax = fig.add_subplot(1, 2, 1)
            if True:
                fig.suptitle('SLM(x)')
                phasedplotv2(ax,xs,amps,kind='x',xunit=self.xunit) 
            ax = fig.add_subplot(1, 2, 2)
            fig.suptitle('SLM(x)')
            plotv2(ax,xs,np.angle(amps),kind='phi',xunit=self.xunit)
            if save :
                plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
                fig.savefig(f'{filename}.svg', format='svg', dpi=1200)

    def plotwpxy(self,xs,ys,nofbins=300,rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False,show3d=False):
        if ys.size<3:
            miny=np.min(ys) 
            maxy=np.max(ys) 
            if symmetrize:
                maxy=max(abs(miny),abs(maxy))
                miny=-maxy
            
            ys=np.linspace(miny,maxy,nofbins)

        if xs.size<3:
            minx=np.min(xs) 
            maxx=np.max(xs) 
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofbins)
           
        rescalesign= -1 if paritytransform else 1  
        coefxy=abs(rescaleamp)*rescalesign
        amps=rescaleamp*self.wpxy(xs=xs*coefxy,ys=ys*coefxy)
        if onlygivedata:
            return xs,ys,amps
        else:
            title='SLM(x,y)'
            phasedonlyplot(xs,ys,amps,xunit=self.xunit,title=title)
            if show3d:
                fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
                allphased3dplots(fig,xs,ys,amps,xunit=self.xunit,title=title)
    def savedensitieswp(self,gridd,gridv,xs,ys,griddname='0',nofbins=300,norm=1,cbarsize=.05,cbardirection='x',offsetration=.2):
        gridd.name=f'density'
        gridv.name='temperature'
        xps,yps, ampxys =self.plotwpxy(xs=xs,ys=ys,nofbins=nofbins,onlygivedata=True)
        
        abampxys=abs(ampxys)
        norm=norm/np.max(abampxys)
        abampxys =norm* abampxys.reshape((ampxys.shape[0], ampxys.shape[1], 1))

        Y, X = np.meshgrid(yps,xps)
        
        phs=0.5+np.angle(ampxys)/(2*np.pi)
        print(np.min(phs),np.max(phs),'are between 0 and 1')
        
        phasecolval=phs.reshape((ampxys.shape[0], ampxys.shape[1], 1))
        if cbarsize!=0:
            w = ampxys.shape[0] if cbardirection=='y' else ampxys.shape[1]
            l = ampxys.shape[1] if cbardirection=='y' else ampxys.shape[0]
            l-=1

            b=int(w-1-cbarsize*w) if cbardirection=='y' else 0
            e=w-1 if cbardirection=='y' else int(cbarsize*w)
            offset=int(l*offsetration)
            l=l-2*offset
            for ll in range(l):
                if cbardirection=='y':
                    phasecolval[b:e,offset+ ll,0]=1.0* ll/l
                    abampxys[b:e,offset+ll,0]=1
                else:
                    phasecolval[offset+ ll,b:e,0]=1.0* ll/l
                    abampxys[offset+ll,b:e,0]=1

            if cbardirection=='y':
                phasecolval[b:e,offset+l,0]=1
            else:
                phasecolval[offset+l,b:e,0]=1

   
        gridd.copyFromArray(abampxys)
        gridv.copyFromArray(phasecolval)

        

        return xps,yps     
    
    def modulate(self, ingbeams):
        outbeams=np.empty([self.amps.size*ingbeams.size],dtype=gbeam.GBeam)
        outind=0
        for inbeam in ingbeams:
            for slmind in range(self.amps.size):
                outbeams[outind]=copy.deepcopy(inbeam)
                if self.ks.ndim==1:
                    outbeams[outind].rotate(self.amps[slmind],dkx=self.ks[slmind],dky=0)
                else: 
                    outbeams[outind].rotate(self.amps[slmind],dkx=self.ks[slmind][0],dky=self.ks[slmind][1])
                outind+=1
        return outbeams
class SLM2d:
    def __init__(self,amps,ks):
        self.size=len(amps[0])*len(amps[1])
        self.amps=np.empty(self.size,dtype=type(amps[0][0]))
        self.ks=np.empty([2,self.size],dtype=float)
        kk=-1
        for ii in range (amps[0].size):
            for jj in range(amps[1].size):
                kk+=1
                self.amps[kk]=amps[0][ii]*amps[1][jj]
                self.ks[0][kk]=ks[0][ii]
                self.ks[1][kk]=ks[1][jj]

    
def cutampstoview(abamps,rangeratio=.99, zeroratio=.02):
    minamp=np.min(abamps)
    maxamp=np.max(abamps)
    if rangeratio==1:
        return abamps
    else:
        meanamp=np.mean(abamps)
        zero=zeroratio*meanamp
        range=rangeratio*(maxamp-minamp)
        downlimit=np.max([zero,meanamp-range/2])
        uplimit=np.min([maxamp,meanamp+range/2])
        abamps[abamps<downlimit]=0
        abamps[abamps>uplimit]=uplimit
        return abamps

class qfosys: 
    def __init__(self,input,pos,output):
        self.xunit=xunit
        self.input=input
        self.output=output
        self.pos=pos

    def setranges(self, zmin, zmax):
        self.zmin=zmin
        self.zmax=zmax

        if zmin<=self.pos and zmax>=self.pos:
            self.input.setranges(zmin, self.pos)
            self.output.setranges(self.pos,zmax)

            self.minx=min(self.input.minx,self.output.minx)
            self.maxx=max(self.input.maxx,self.output.maxx)
            self.minkx=min(self.input.minkx,self.output.minkx)
            self.maxkx=max(self.input.maxkx,self.output.maxkx)

            self.miny=min(self.input.miny,self.output.miny)
            self.maxy=max(self.input.maxy,self.output.maxy)
            self.minky=min(self.input.minky,self.output.minky)
            self.maxky=max(self.input.maxky,self.output.maxky)

        elif zmin>zmax:
            print("ERROR: zmin>zmax!!!? ")
            sys.exit()
        elif zmin>= self.pos and zmax>=self.pos:
            self.output.setranges(zmin, zmax)

            self.minx=self.output.minx
            self.maxx=self.output.maxx
            self.minkx=self.output.minkx
            self.maxkx=self.output.maxkx

            self.miny=self.output.miny
            self.maxy=self.output.maxy
            self.minky=self.output.minky
            self.maxky=self.output.maxky

        elif zmin<=self.pos and zmax<=self.pos:
            self.input.setranges(zmin, zmax)
            self.minx=self.input.minx
            self.maxx=self.input.maxx
            self.minkx=self.input.minkx
            self.maxkx=self.input.maxkx

            self.miny=self.input.miny
            self.maxy=self.input.maxy
            self.minky=self.input.minky
            self.maxky=self.input.maxky

    def sourceatz(self,z):
        """After using sourceatz, you need to apply goto on the source!"""
        mysource= self.input if z<self.pos else self.output
        return mysource.sourceatz(z)
    def stateatz(self,z):
        mysource= self.sourceatz(z)
        mysource.goto(z)
        return mysource
    def getampsatz(self,z):
        mysource= self.sourceatz(z)
        return mysource.getampsatz(z)
    def printfockrepatz(self,z,rounddig=4,showy=True,absamps=False,projns=[],projxs=[],projys=[],demolish=False):
        mysource= self.sourceatz(z)
        return mysource.printfockrepatz(z,rounddig=rounddig,showy=showy,absamps=absamps,projns=projns,projxs=projxs,projys=projys,demolish=demolish)
    def project(self,z,projns,projxs,projys,demolish=False):
        mysource= self.sourceatz(z)
        return mysource.project(z=z,projns=projns,projxs=projxs,projys=projys,demolish=demolish)
    def intensityxatz(self,z,xs):
        mysource= self.sourceatz(z)
        return mysource.intensityxatz(z,xs)
    def intensitykxatz(self,z,kxs):
        mysource= self.sourceatz(z)
        return mysource.intensitykxatz(z,kxs)

    def wpxatz(self,z,xs):
        mysource= self.sourceatz(z)
        return mysource.wpxatz(z,xs)
    def wpyatz(self,z,ys):
        mysource= self.sourceatz(z)
        return mysource.wpyatz(z,ys)
    def wpkxatz(self,z,kxs):
        mysource= self.sourceatz(z)
        return mysource.wpkxatz(z,kxs)
    def wpkyatz(self,z,kys):
        mysource= self.sourceatz(z)
        return mysource.wpkyatz(z,kys)
    def wpxyatz(self,z,xs,ys):
        mysource= self.sourceatz(z)
        return mysource.wpxyatz(z,xs,ys)
    def wpkxyatz(self,z,kxs,kys):
        mysource= self.sourceatz(z)
        return mysource.wpkxyatz(z,kxs,kys)
    def wpxylinearatz(self,z,xs,ys):
        mysource= self.sourceatz(z)
        return mysource.wpxylinearatz(z,xs,ys)

    def movewpx(self,zs,xs):
        out=np.empty([zs.size,xs.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.wpxatz(zs[iz],xs)
        return out
    
    def moveintensityx(self,zs,xs):
        out=np.empty([zs.size,xs.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.intensityxatz(zs[iz],xs)
        return out
    
    def movewpx2(self,zs,xs): # it gets 2d xs and means over one dimension
        myxs=xs.flatten()
        out=np.empty([zs.size,xs.shape[0]],dtype=complex)
        for iz in range(zs.size):
            preout=self.wpxatz(zs[iz],myxs)
            preout=preout.reshape((xs.shape[0],-1))
            ang=np.mean(np.angle(preout),axis=1)
            amp=np.mean(abs(preout),axis=1)
            out[iz,:]=amp*np.exp(1j*ang)
        return out
    
    def moveintensityx2(self,zs,xs): #it gets 2d xs and means over one dimension
        myxs=xs.flatten()
        out=np.empty([zs.size,xs.shape[0]],dtype=complex)
        for iz in range(zs.size):
            preout=self.intensityxatz(zs[iz],myxs)
            preout=preout.reshape((xs.shape[0],-1))
            amp=np.mean(preout,axis=1)
            out[iz,:]=amp
        return out
        
    def movewpkx(self,zs,kxs):
        out=np.empty([zs.size,kxs.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.wpkxatz(zs[iz],kxs)
        return out
    def moveintensitykx(self,zs,kxs):
        out=np.empty([zs.size,kxs.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.intensitykxatz(zs[iz],kxs)
        return out
    def movewpy(self,zs,ys):
        out=np.empty([zs.size,ys.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.wpyatz(zs[iz],ys)
        return out
    def movewpky(self,zs,kys):
        out=np.empty([zs.size,kys.size],dtype=complex)
        for iz in range(zs.size): 
            out[iz,:]=self.wpkyatz(zs[iz],kys)
        return out
    
    def plotintensityxatz(self,z,nofxbins=300,xs=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        mysource= self.sourceatz(z)
        return mysource.plotintensityxatz(z=z,nofxbins=nofxbins,xs=xs,rescaleamp=rescaleamp,paritytransform=paritytransform,symmetrize=symmetrize,onlygivedata=onlygivedata,xunit=self.xunit)
    def plotintensityxatz000(self,z,nofxbins=300,xs=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        if xs.size<3:
            minx=np.min(xs) if xs.size==2 else self.minx
            maxx=np.max(xs) if xs.size==2 else self.maxx
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofxbins)

        rescalesign= -1 if paritytransform else 1  
        amps=rescaleamp*self.intensityxatz(z,xs*abs(rescaleamp)**2*rescalesign) # it is for FT and convolution test
        if onlygivedata:
            return xs, amps
        else:
            fig = plt.figure() 
            ax = fig.add_subplot(1, 1, 1)
            fig.suptitle(' intensity x at z = '+str(z)+" "+self.xunit, fontsize=16)
            phasedplotv2(ax,xs,amps,kind='x',xunit=self.xunit) 

    def plotwpxatz(self,z,nofxbins=300,xs=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        if xs.size<3:
            minx=np.min(xs) if xs.size==2 else self.minx
            maxx=np.max(xs) if xs.size==2 else self.maxx
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofxbins)
        # int{dx |xi(x)|^2}             =1
        # int{dx |sqrt(a) xi(a x)|^2}   =1
        # int{dx |amp xi(|amp|^2 x)|^2} =1 
        # a= |amp|^2  or amp =sqrt(a)
        #xps*=abs(rescaleamp)**2
        rescalesign= -1 if paritytransform else 1  
        amps=rescaleamp*self.wpxatz(z,xs*abs(rescaleamp)**2*rescalesign) # it is for FT and convolution test
        if onlygivedata:
            return xs, amps
        else:
            fig = plt.figure() 
            ax = fig.add_subplot(1, 1, 1)
            fig.suptitle('Photon-wavepacket x at z = '+str(z)+" "+self.xunit, fontsize=16)
            phasedplotv2(ax,xs,amps,kind='x',xunit=self.xunit) 

    def plotwpyatz(self,z,nofybins=300,ys=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        if ys.size<3:
            miny=np.min(ys) if ys.size==2 else self.miny
            maxy=np.max(ys) if ys.size==2 else self.maxy
            if symmetrize:
                maxy=max(abs(miny),abs(maxy))
                miny=-maxy
            ys=np.linspace(miny,maxy,nofybins)
        rescalesign= -1 if paritytransform else 1  
        amps=rescaleamp*self.wpyatz(z,ys*abs(rescaleamp)**2*rescalesign) # it is for FT and convolution test
        if onlygivedata:
            return ys, amps
        else:
            fig = plt.figure()  
            ax = fig.add_subplot(1, 1, 1)
            
            fig.suptitle('Photon-wavepacket y at z = '+str(z)+" "+self.xunit, fontsize=16)
            phasedplotv2(ax,ys,amps,kind='y',xunit=self.xunit)  
    def plotwpkxatz(self,z,nofxbins=300,kxs=np.array([-1]),rescaleamp=1,symmetrize=False,onlygivedata=False):
        if kxs.size<3:
            minkx=np.min(kxs) if kxs.size==2 else self.minkx
            maxkx=np.max(kxs) if kxs.size==2 else self.maxkx
            if symmetrize:
                maxkx=max(abs(minkx),abs(maxkx))
                minkx=-maxkx
            dkx=(maxkx-minkx)/nofxbins
            maxkx+=dkx/3
            kxs=np.arange(minkx,maxkx,dkx)
        amps=rescaleamp*self.wpkxatz(z,kxs*abs(rescaleamp)**2) # it is for FT and convolution test

        if onlygivedata:
            return kxs, amps
        else:
            fig = plt.figure()  
            ax = fig.add_subplot(1, 1, 1)
            phasedplotv2(ax,kxs,amps,kind='k_x',xunit=self.xunit)
     
    def plotwpkyatz(self,z,nofybins=300,kys=np.array([-1]),rescaleamp=1,symmetrize=False,onlygivedata=False):
        if kys.size<3:
            minky=np.min(kys) if kys.size==2 else self.minky
            maxky=np.max(kys) if kys.size==2 else self.maxky
            if symmetrize:
                maxky=max(abs(minky),abs(maxky))
                minky=-maxky
            dky=(maxky-minky)/nofybins
            maxky+=dky/3
            kys=np.arange(minky,maxky,dky)
        amps=rescaleamp*self.wpkyatz(z,kys*abs(rescaleamp)**2 ) # it is for FT and convolution test
        if onlygivedata:
            return kys, amps
        else:
            fig = plt.figure()  
            ax = fig.add_subplot(1, 1, 1)
            phasedplotv2(ax,kys,amps,kind='k_y',xunit=self.xunit)  
    
    def plotwpxyatz(self,z,nofbins=300,xs=np.array([-1]),ys=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        if ys.size<3:
            miny=np.min(ys) if ys.size==2 else self.miny
            maxy=np.max(ys) if ys.size==2 else self.maxy
            if symmetrize:
                maxy=max(abs(miny),abs(maxy))
                miny=-maxy
            
            ys=np.linspace(miny,maxy,nofbins)

        if xs.size<3:
            minx=np.min(xs) if xs.size==2 else self.minx
            maxx=np.max(xs) if xs.size==2 else self.maxx
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofbins)
        rescalesign= -1 if paritytransform else 1  
        coefxy=abs(rescaleamp)*rescalesign
        amps=rescaleamp*self.wpxyatz(z,xs=xs*coefxy,ys=ys*coefxy) # it is for FT and convolution test
        if onlygivedata:
            return xs,ys,amps
        else:
            title=rf'Photon-wavepacket $\xi(x,y)$ at $z$ =  {z:.3f}'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,xs,ys,amps,xunit=self.xunit,title=title)
            
    
    def plotwpkxyatz(self,z,nofbins=300,kxs=np.array([-1]),kys=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False):
        
        if kys.size<3:
            minky=np.min(kys) if kys.size==2 else self.minky
            maxky=np.max(kys) if kys.size==2 else self.maxky
            if symmetrize:
                maxky=max(abs(minky),abs(maxky))
                minky=-maxky
            dky=(maxky-minky)/nofbins
            maxky+=dky/30
            kys=np.arange(minky,maxky,dky)
        if kxs.size<3:
            minkx=np.min(kxs) if kxs.size==2 else self.minkx
            maxkx=np.max(kxs) if kxs.size==2 else self.maxkx
            if symmetrize:
                maxkx=max(abs(minkx),abs(maxkx))
                minkx=-maxkx
            dkx=(maxkx-minkx)/nofbins
            maxkx+=dkx/30
            kxs=np.arange(minkx,maxkx,dkx)
        rescalesign= -1 if paritytransform else 1  
        coefxy=abs(rescaleamp)*rescalesign
        amps=rescaleamp*self.wpkxyatz(z,kxs=kxs*coefxy,kys=kys*coefxy) # it is for FT and convolution test
        if onlygivedata:
            return kxs,kys,amps
        else:
            title=rf'Photon-wavepacket $\tilde{{\xi}}(k_x,k_y)$ at $z$ =  {z:.3f}'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,kxs,kys,amps,xunit=self.xunit,title=title)
    
    def plotxwpsatz(self,z,nofxbins=300,xs=np.array([-1]),kxs=np.array([-1])):
        kxps, ampkxs=self.plotwpkxatz(z,nofxbins,kxs,onlygivedata=True)
        xps, ampxs =self.plotwpxatz(z,nofxbins,xs,onlygivedata=True)
       
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle('Photon-wavepacket x at z = '+str(z)+" "+self.xunit, fontsize=16)
        phasedplotv2(axs[0],xps,ampxs,kind='x',xunit=self.xunit) 
        phasedplotv2(axs[1],kxps,ampkxs,kind='k_x',xunit=self.xunit)  
    
    def plotywpsatz(self,z,nofybins=300,ys=np.array([-1]),kys=np.array([-1]),rescaleampy=1,paritytransformy=False):
        kyps, ampkys=self.plotwpkyatz(z,nofybins,kys,onlygivedata=True)
        yps, ampys =self.plotwpyatz(z,nofybins,ys,onlygivedata=True)
       
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle('Photon-wavepacket y at z = '+str(z)+" "+self.xunit, fontsize=16)
        phasedplotv2(axs[0],yps,ampys,kind='y',xunit=self.xunit) 
        phasedplotv2(axs[1],kyps,ampkys,kind='k_y',xunit=self.xunit)  

    def plotwpsatz(self,z,nofbins=300,xs=np.array([-1]),ys=np.array([-1]),kxs=np.array([-1]),kys=np.array([-1])):
        kxps,kyps, ampkxys=self.plotwpkxyatz(z,nofbins,kxs=kxs,kys=kys,onlygivedata=True)
        
        xps,yps, ampxys =self.plotwpxyatz(z,nofbins,xs=xs,ys=ys,onlygivedata=True)
        fig = plt.figure(figsize=plt.figaspect(2/3), constrained_layout=True) 
        subfigs = fig.subfigures(nrows=2, ncols=1)
        title=rf'Photon-wavepacket $\xi(x,y)$ at $z$ =  {z:.3f}'+self.xunit
        allphased3dplots(subfigs[0],xps,yps,ampxys,kind='r',xunit=self.xunit,title=title)
        title=rf'Photon-wavepacket $\tilde{{\xi}}(k_x,k_y)$ at $z$ =  {z:.3f}'+self.xunit
        allphased3dplots(subfigs[1],kxps,kyps,ampkxys,kind='k',xunit=self.xunit,title=title)

    def plotintensityx(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False,returnnonmeanxs=False):
        minz=np.min(zs) if zs.size>1 else self.zmin
        maxz=np.max(zs) if zs.size>1 else self.zmax 
        minx=np.min(xs) if xs.size>1 else self.minx
        maxx=np.max(xs) if xs.size>1 else self.maxx
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        if xs.size<3:
            self.setranges(minz,maxz)
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofxbins*nofxsubbins)
            if nofxsubbins!= 1: 
                xs=xs.reshape((nofxbins,nofxsubbins))
        
        if xs.size!= xs.shape[0]:#nofxsubbins!= 1: 
            amps=self.moveintensityx2(zs,xs)#self.movewpx2(zs,xs)
            meanxs=np.mean(xs,axis=1)
        else:
            amps=self.moveintensityx(zs,xs)#self.movewpx(zs,xs)
            meanxs=xs
        
        if onlygivedata:
            if returnnonmeanxs:#nofxsubbins!= 1: 
                return zs,meanxs,amps,xs
            else:
                return zs,meanxs,amps
        else:
            title=rf'Photon-intensity $I(x)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplotsvs(fig,zs,meanxs,amps,kind='r',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)
    
    def plotintensitykx(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,kxs=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False):
        minz=np.min(zs) if zs.size>1 else self.zmin 
        maxz=np.max(zs) if zs.size>1 else self.zmax
        minkx=np.min(kxs) if kxs.size>1 else self.minkx
        maxkx=np.max(kxs) if kxs.size>1 else self.maxkx
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        if kxs.size<3:
            self.setranges(minz,maxz)
            if symmetrize:
                maxkx=max(abs(minkx),abs(maxkx))
                minkx=-maxkx
            dx=(maxkx-minkx)/nofxbins
            maxkx+=dx/3
            kxs=np.arange(minkx,maxkx,dx)
        amps=self.moveintensitykx(zs,kxs)
        if onlygivedata:
            return zs,kxs,amps
        else:
            title=rf'Photon-intensity $I(k_x)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,zs,kxs,amps,kind='k',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)

    def plotwpIx(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False,returnnonmeanxs=False,kind='wp'):
        minz=np.min(zs) if zs.size>1 else self.zmin
        maxz=np.max(zs) if zs.size>1 else self.zmax 
        minx=np.min(xs) if xs.size>1 else self.minx
        maxx=np.max(xs) if xs.size>1 else self.maxx
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        if xs.size<3:
            self.setranges(minz,maxz)
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofxbins*nofxsubbins)
            if nofxsubbins!= 1: 
                xs=xs.reshape((nofxbins,nofxsubbins))
        
        if xs.size!= xs.shape[0]:#nofxsubbins!= 1: 
            if kind=='wp':
                amps=self.movewpx2(zs,xs)
            elif kind=='I':
                amps=self.moveintensityx2(zs,xs)
            meanxs=np.mean(xs,axis=1)
        else:
            if kind=='wp':
                amps=self.movewpx(zs,xs)
            elif kind=='I':
                amps=self.moveintensityx(zs,xs)
            meanxs=xs
        
        if onlygivedata:
            if returnnonmeanxs:#nofxsubbins!= 1: 
                return zs,meanxs,amps,xs
            else:
                return zs,meanxs,amps
        else:
            title=rf'Photon-wavepacket $\xi(x)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,zs,meanxs,amps,kind='r',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)

    def plotwpkx(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,kxs=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False):
        minz=np.min(zs) if zs.size>1 else self.zmin 
        maxz=np.max(zs) if zs.size>1 else self.zmax
        minkx=np.min(kxs) if kxs.size>1 else self.minkx
        maxkx=np.max(kxs) if kxs.size>1 else self.maxkx
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        if kxs.size<3:
            self.setranges(minz,maxz)
            if symmetrize:
                maxkx=max(abs(minkx),abs(maxkx))
                minkx=-maxkx
            dx=(maxkx-minkx)/nofxbins
            maxkx+=dx/3
            kxs=np.arange(minkx,maxkx,dx)
        amps=self.movewpkx(zs,kxs)
        if onlygivedata:
            return zs,kxs,amps
        else:
            title=rf'Photon-wavepacket $\tilde \xi(k_x)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,zs,kxs,amps,kind='k',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)
    
    def plotwpy(self,nofzbins=300,zs=np.array([-1]),nofybins=300,ys=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False):
        minz=np.min(zs) if zs.size>1 else self.zmin
        maxz=np.max(zs) if zs.size>1 else self.zmax
        miny=np.min(ys) if ys.size>1 else self.miny
        maxy=np.max(ys) if ys.size>1 else self.maxy
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        if ys.size<3:
            self.setranges(minz,maxz)
            if symmetrize:
                maxy=max(abs(miny),abs(maxy))
                miny=-maxy
            ys=np.linspace(miny,maxy,nofybins)
        amps=self.movewpy(zs,ys)
        if onlygivedata:
            return zs,ys,amps
        else:
            title=rf'Photon-wavepacket $\xi(y)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,zs,ys,amps,kind='r',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)
    
    def plotwpky(self,nofzbins=300,zs=np.array([-1]),nofybins=300,kys=np.array([-1]),symmetrize=False,onlygivedata=False,cuttoview=False):
        minz=np.min(zs) if zs.size>1 else self.zmin
        maxz=np.max(zs) if zs.size>1 else self.zmax
        minky=np.min(kys) if kys.size>1 else self.minky
        maxky=np.max(kys) if kys.size>1 else self.maxky
        if zs.size<3:
            zs=np.linspace(minz,maxz,nofzbins)
        self.setranges(minz,maxz)
        if kys.size<3:
            if symmetrize:
                maxky=max(abs(minky),abs(maxky))
                minky=-maxky
            dy=(maxky-minky)/nofybins
            maxky+=dy/3
            kys=np.arange(minky,maxky,dy)
        amps=self.movewpkx(zs,kys)
        if onlygivedata:
            return zs,kys,amps
        else:
            title=rf'Photon-wavepacket $\tilde \xi(k_y)$ for z = ({min(zs):.2f},{max(zs):.2f})'+self.xunit
            fig = plt.figure(figsize=plt.figaspect(1/3), constrained_layout=True) 
            allphased3dplots(fig,zs,kys,amps,kind='k',xunit=self.xunit,title=title,squarefig=False,cuttoview=cuttoview)

    def plotxwps(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),kxs=np.array([-1]),symmetrize=False,cuttoview=False):
        zps1,kxps,ampkxs=self.plotwpkx(nofzbins,zs,nofxbins,kxs,symmetrize,onlygivedata=True)
        zps2,xps, ampxs =self.plotwpIx(nofzbins=nofzbins,zs=zs,nofxbins=nofxbins,nofxsubbins=nofxsubbins,xs=xs,symmetrize=symmetrize,onlygivedata=True,cuttoview=False)
        
        fig = plt.figure(figsize=plt.figaspect(2/3), constrained_layout=True) 
        subfigs = fig.subfigures(nrows=2, ncols=1)
        title=rf'Photon-wavepacket $\xi(x)$ for z = ({min(zps1):.2f},{max(zps1):.2f})'+self.xunit
        allphased3dplots(subfigs[0],zps1,xps,ampxs,kind='zx',xunit=self.xunit,title=title,cuttoview=cuttoview)
        title=rf'Photon-wavepacket $\tilde \xi(k_x)$ for z = ({min(zps2):.2f},{max(zps2):.2f})'+self.xunit
        allphased3dplots(subfigs[1],zps2,kxps,ampkxs,kind='zkx',xunit=self.xunit,title=title,cuttoview=cuttoview)
    
    
    def plotywps(self,nofzbins=300,zs=np.array([-1]),nofybins=300,ys=np.array([-1]),kys=np.array([-1]),symmetrize=False,cuttoview=False):
        zps1,kyps,ampkys=self.plotwpky(nofzbins,zs,nofybins,kys,symmetrize,onlygivedata=True)
        zps2,yps, ampys =self.plotwpy(nofzbins,zs,nofybins,ys,symmetrize,onlygivedata=True)
        
        fig = plt.figure(figsize=plt.figaspect(2/3), constrained_layout=True) 
        subfigs = fig.subfigures(nrows=2, ncols=1)
        title=rf'Photon-wavepacket $\xi(y)$ for z = ({min(zps1):.2f},{max(zps1):.2f})'+self.xunit
        allphased3dplots(subfigs[0],zps1,yps,ampys,kind='zy',xunit=self.xunit,title=title,cuttoview=cuttoview)
        title=rf'Photon-wavepacket $\tilde \xi(k_y)$ for z = ({min(zps2):.2f},{max(zps2):.2f})'+self.xunit
        allphased3dplots(subfigs[1],zps2,kyps,ampkys,kind='zky',xunit=self.xunit,title=title,cuttoview=cuttoview)
    
    def plotxintensities(self,nofzbins=300,zs=np.array([-1]),nofxbins=300,nofxsubbins=1,xs=np.array([-1]),kxs=np.array([-1]),symmetrize=False,cuttoview=False):
        zps1,kxps,ampkxs=self.plotintensitykx(nofzbins,zs,nofxbins,kxs,symmetrize,onlygivedata=True)
        zps2,xps, ampxs =self.plotintensityx(nofzbins=nofzbins,zs=zs,nofxbins=nofxbins,nofxsubbins=nofxsubbins,xs=xs,symmetrize=symmetrize,onlygivedata=True,cuttoview=False)
        
        fig = plt.figure(figsize=plt.figaspect(2/3), constrained_layout=True) 
        subfigs = fig.subfigures(nrows=2, ncols=1)
        title=f'intensity x for z = ({min(zps1):.2f},{max(zps1):.2f})'+self.xunit
        allphased3dplots(subfigs[0],zps1,xps,ampxs,kind='r',xunit=self.xunit,title=title,cuttoview=cuttoview)
        title=f'intensity kx for z = ({min(zps2):.2f},{max(zps2):.2f})'+self.xunit
        allphased3dplots(subfigs[1],zps2,kxps,ampkxs,kind='k',xunit=self.xunit,title=title,cuttoview=cuttoview)
    
            #savedensitieswpx
    def savedensitieswpIx(self,gridd,griddname='densitywpx',normval=10,nofzbins=500,zs=np.array([-1]),nofxbins=700,nofxsubbins=1,xs=np.array([-1]),symmetrize=False,cuttoview=False,kind='wp',normalize=1,projz=4,includephase=False,gridv=0):
        if includephase:
            gridd.name=f'density'
            gridv.name='temperature'
        else:
            gridd.name = griddname
        
        zps,meanxs,ampxs,xps =self.plotwpIx(nofzbins=nofzbins,zs=zs,nofxbins=nofxbins,nofxsubbins=nofxsubbins,xs=xs,symmetrize=symmetrize,onlygivedata=True,cuttoview=False,returnnonmeanxs=True,kind=kind)
        if kind=='wp' and not includephase:
            abampxs=abs(ampxs)**2 # density=|wp|^2
            abampisintensity=True
        elif kind=='wp' and includephase:
            abampxs=abs(ampxs)   # when we plot phase, it is better to consider amplitude but not intensity for the brightness
            abampisintensity=False
            phs=0.5+np.angle(ampxs)/(2*np.pi)
            print(np.min(phs),np.max(phs),'are between 0 and 1')
            phasecolval=phs.reshape((ampxs.shape[0], ampxs.shape[1], 1))
        elif kind=='I' and not includephase:
            abampxs=abs(ampxs) # density=|I|
            abampisintensity=True

        elif kind=='I' and includephase:
            print('we can not extract the phase from the intensity')
            err=1/0
        else:
            err=1/0
        
        if cuttoview:
            abampxs=cutampstoview(abampxs,rangeratio=.99, zeroratio=.02) # make rangeration=1 fo no cutting
        if normalize>0:    
            for iz in range(zps.size): 
                if abampisintensity :
                    nfactor=normval/(np.sum(abampxs[iz,:]))
                else:
                    nfactor=normval/(np.sum(abampxs[iz,:]**2))
                abampxs[iz,:]*=nfactor
                if normalize==123:
                    if zps[iz]>projz:
                        abampxs[iz,:]*=.11
        abampxs = abampxs.reshape((abampxs.shape[0], abampxs.shape[1], 1))
        gridd.copyFromArray(abampxs)
        if includephase:
            gridv.copyFromArray(phasecolval)


        voxelx=.01
        voxely=.01
        voxelz=.01
        tranmatrix=[[voxelx, 0, 0, 0],[0, voxely, 0, 0],[0, 0, voxelz, 0],[-8.08, -2.8, -9.34, 1]]
        print('minx=',np.min(meanxs),'maxx=',np.max(meanxs))

        return zps,xps
    def savedensitieswpsatz(self,z,gridd,gridv,griddname='0',nofbins=300,xs=np.array([-1]),ys=np.array([-1]),norm=1,cbarsize=.05,cbardirection='x',offsetration=.2):
        gridd.name = f'densityatz={z}'
        gridd.name=f'density={z}'
        gridd.name=f'density'
        gridv.name='temperature'
        xps,yps, ampxys =self.plotwpxyatz(z,nofbins,xs=xs,ys=ys,onlygivedata=True)
        abampxys=abs(ampxys)
        norm=norm/np.max(abampxys)
        abampxys =norm* abampxys.reshape((ampxys.shape[0], ampxys.shape[1], 1))

        Y, X = np.meshgrid(yps,xps)
        print(xps.shape,'tst',yps.shape,ampxys.shape,Y.shape,X.shape)
        X=X.flatten()
        X=X/max(X)
        Y=Y.flatten()
        Y=Y/max(Y)
        
        mx=max(X)/4
        X=np.remainder(X, mx)/mx
        phs=0.5+np.angle(ampxys)/(2*np.pi)
        print(np.min(phs),np.max(phs),'are between 0 and 1')
        
        phasecolval=phs.reshape((ampxys.shape[0], ampxys.shape[1], 1))
        if cbarsize!=0:
            w = ampxys.shape[0] if cbardirection=='y' else ampxys.shape[1]
            l = ampxys.shape[1] if cbardirection=='y' else ampxys.shape[0]
            l-=1

            b=int(w-1-cbarsize*w) if cbardirection=='y' else 0
            e=w-1 if cbardirection=='y' else int(cbarsize*w)
            offset=int(l*offsetration)
            l=l-2*offset
            for ll in range(l):
                if cbardirection=='y':
                    phasecolval[b:e,offset+ ll,0]=1.0* ll/l
                    abampxys[b:e,offset+ll,0]=1
                else:
                    phasecolval[offset+ ll,b:e,0]=1.0* ll/l
                    abampxys[offset+ll,b:e,0]=1

            if cbardirection=='y':
                phasecolval[b:e,offset+l,0]=1
            else:
                phasecolval[offset+l,b:e,0]=1
        gridd.copyFromArray(abampxys)
        gridv.copyFromArray(phasecolval)
 
        return xps,yps

    def savedensitieswpsatz3d(self,z,gridd,gridv,griddname='0',nofbins=300,nofzbins=300,xs=np.array([-1]),ys=np.array([-1]),norm=1,cbarsize=.05,cbardirection='x',offsetration=.2):
        gridd.name = f'densityatz={z}'
        gridd.name=f'density={z}'
        gridd.name=f'density'
        gridv.name='temperature'
        xps,yps, ampxys =self.plotwpxyatz(z,nofbins,xs=xs,ys=ys,onlygivedata=True)
        abampxys=abs(ampxys)
        norm=norm/np.max(abampxys)
        abampxys*=norm
        phs=0.5+np.angle(ampxys)/(2*np.pi)
        outabamp = np.zeros((nofbins,nofbins,nofzbins),dtype=np.float)
        outphs = np.zeros((nofbins,nofbins,nofzbins),dtype=np.float)
        
        for ii in range(ampxys.shape[0]):
            for jj in range(ampxys.shape[1]):
                amp=abampxys[ii,jj]
                kk=int(amp*(nofzbins-1))
                outabamp[ii,jj,kk]=amp
                outphs[ii,jj,kk]=phs[ii,jj]

        abampxys =abampxys.reshape((ampxys.shape[0], ampxys.shape[1], 1))
       
       
        phasecolval=phs.reshape((ampxys.shape[0], ampxys.shape[1], 1))
        if cbarsize!=0:
            w = ampxys.shape[0] if cbardirection=='y' else ampxys.shape[1]
            l = ampxys.shape[1] if cbardirection=='y' else ampxys.shape[0]
            l-=1

            b=int(w-1-cbarsize*w) if cbardirection=='y' else 0
            e=w-1 if cbardirection=='y' else int(cbarsize*w)
            offset=int(l*offsetration)
            l=l-2*offset
            for ll in range(l):
                if cbardirection=='y':
                    phasecolval[b:e,offset+ ll,0]=1.0* ll/l
                    abampxys[b:e,offset+ll,0]=1
                else:
                    phasecolval[offset+ ll,b:e,0]=1.0* ll/l
                    abampxys[offset+ll,b:e,0]=1

            if cbardirection=='y':
                phasecolval[b:e,offset+l,0]=1
            else:
                phasecolval[offset+l,b:e,0]=1

   
        gridd.copyFromArray(outabamp)
        gridv.copyFromArray(outphs)
        return xps,yps

    def savedensities(self,gridd,griddname='density',norm=10,maxpointsperpage=1000,zlim=np.array([-1]),xlim=np.array([-1]),ylim=np.array([-1]),symmetrize=False,maxbinsperline=2000,zerolimit=.001):
        gridd.name = griddname
        dAccessor = gridd.getAccessor()

        minz=np.min(zlim) if zlim.size==2 else self.zmin
        maxz=np.max(zlim) if zlim.size==2 else self.zmax
        self.setranges(minz,maxz)
        
        minx=np.min(xlim) if xlim.size==2 else self.minx
        maxx=np.max(xlim) if xlim.size==2 else self.maxx
        miny=np.min(ylim) if ylim.size==2 else self.miny
        maxy=np.max(ylim) if ylim.size==2 else self.maxy
        if symmetrize:
            maxx=max(abs(minx),abs(maxx))
            minx=-maxx 
            maxy=max(abs(miny),abs(maxy))
            miny=-maxy 
        Dx=maxx-minx
        Dy=maxy-miny
        Dz=maxz-minz
        ddx=Dx/maxbinsperline
        ddy=Dx/maxbinsperline
        ddz=Dx/maxbinsperline
        zs=np.arange(minz,maxz+ddz/3,ddz)
        xs=np.arange(minx,maxx+ddx/3,ddx)
        ys=np.arange(miny,maxy+ddy/3,ddz)
        znofbins=zs.size
        valarr = np.ndarray((xs.size, ys.size))
        myzerolimitforpoints=zerolimit/maxpointsperpage
        for k in range(znofbins):
            self.setranges(zmin=zs[k],zmax=zs[k])
            myminx=max(minx,self.minx)
            mymaxx=min(maxx,self.maxx)
            myminy=max(minx,self.miny)
            mymaxy=min(maxy,self.maxy)
            p=0
            n=0
            myxs=np.random.uniform(myminx,mymaxx,maxpointsperpage)
            myys=np.random.uniform(myminy,mymaxy,maxpointsperpage)
            myamps=abs(self.wpxylinearatz(z=zs[k],xs=myxs,ys=myys))**2
            myamps[myamps<myzerolimitforpoints*(np.sum(myamps))]=0
            nfactor=norm/(np.sum(myamps))
            myamps*=nfactor
            myis=np.round((myxs-minx)/ddx).astype(int)
            myjs=np.round((myys-miny)/ddy).astype(int)
            valarr.fill(0)
            for p in range(maxpointsperpage):
                if myamps[p]>myzerolimitforpoints:
                    i=myis[p].item()
                    j=myjs[p].item()
                    ijk = (k,i,j)
                    valarr[i,j]+=myamps[p]
                    valarr[myis[p],myjs[p]]+=myamps[p]
                    dAccessor.setValueOn(ijk,valarr[i,j])
        del dAccessor
        return xs,ys,zs
    
class lenssys(qfosys):
    def __init__(self,insource,lens,lenspos=0):
        self.lens= lens
        self.longphase=insource.longphase
        myinsource=copy.deepcopy(insource)
        myoutsource=copy.deepcopy(insource)
        myoutsource.goto(lenspos)
        myoutsource.passlens(lens)
        qfosys.__init__(self,input=myinsource,pos=lenspos,output=myoutsource)
        self.setranges(zmin=lenspos-lens.f,zmax=lenspos+lens.f)

class Ffsys(qfosys):#four f
    def __init__(self,insource,lens1,slm,lens2,slmpos=0):
        self.lens1pos=slmpos-lens1.f
        self.lens2pos=slmpos+lens2.f
        self.pos=slmpos
        self.longphase=insource.longphase
        self.slm=slm
        lenssys1=lenssys(insource=insource,lens=lens1,lenspos=self.lens1pos)
        midsource=copy.deepcopy(lenssys1.sourceatz(z=slmpos))
        midsource.goto(slmpos)
        midsource.passslm(slm)

        lenssys2=lenssys(insource=midsource,lens=lens2,lenspos=self.lens2pos)
        ################################################################
        qfosys.__init__(self,input=lenssys1,pos=slmpos,output=lenssys2)
        self.setranges(zmin=slmpos-2*lens1.f,zmax=slmpos+2*lens2.f)
        
    def lensatz(self,z):
        return self.input if z<self.pos else self.output
class Efsys(qfosys):  #eight f
    def __init__(self,insource,lensa1,slma,lensa2,lensb1,slmb,lensb2,slm,slmpos=0):
        slmapos=slmpos-2*lensa2.f
        slmbpos=slmpos+2*lensb1.f
        self.pos=slmpos
        self.longphase=insource.longphase
        self.slm=slm
        Ffsysa=Ffsys(insource=insource,lens1=lensa1,slm=slma,lens2=lensa2,slmpos=slmapos)
        midsource=copy.deepcopy(Ffsysa.sourceatz(z=slmpos))
        midsource.goto(slmpos)
        if isinstance(slm,SLM):
            midsource.passslm(slm)
        else: #isinstance(slm, SLMpix):
            midsource.passslmpix(slm)

        Ffsysb=Ffsys(insource=midsource,lens1=lensb1,slm=slmb,lens2=lensb2,slmpos=slmbpos)
        ################################################################
        qfosys.__init__(self,input=Ffsysa,pos=slmpos,output=Ffsysb)
        self.setranges(zmin=slmapos-2*lensa1.f,zmax=slmbpos+2*lensb2.f)
        
    def Ffatz(self,z):
        return self.input if z<self.pos else self.output 

class EfEProjsys(qfosys):  #eight f projected at end
    def __init__(self,insource,lensa1,slma,lensa2,lensb1,slmb,lensb2,slm,slmpos,projns,projxs,projys,demolish):
        self.longphase=insource.longphase
        slmapos=slmpos-2*lensa2.f
        slmbpos=slmpos+2*lensb1.f
        zend=slmbpos+2*lensb2.f
        self.pos=zend
        theEfsys=Efsys(insource=insource,lensa1=lensa1,slma=slma,lensa2=lensa2,lensb1=lensb1,slmb=slmb,lensb2=lensb2,slm=slm,slmpos=slmpos)
        midsource=copy.deepcopy(theEfsys.sourceatz(z=zend))
        midsource.goto(zend)
        if isinstance(midsource,ESource):
            midsourceE=midsource
        elif isinstance(midsource,FSource):
            midsourceE=ESource(fsources=[midsource],amps=[1],ExpandSimplify=False)
        else: #isinstance(slm, SLMpix):
            err=1/0
        midsourceE.project(z=zend,projns=projns,projxs=projxs,projys=projys,demolish=demolish)

        ################################################################
        qfosys.__init__(self,input=theEfsys,pos=zend,output=midsourceE)
        self.setranges(zmin=slmapos-2*lensa1.f,zmax=zend+lensb2.f)

class nf(qfosys): #eight f
    def __init__(self,input,lens1,slm,lens2,slmpos=0,longphase=True):
        pass 