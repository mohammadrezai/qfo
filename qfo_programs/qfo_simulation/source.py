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

from gbeam import *
import copy
import matplotlib.pyplot as plt
from plot import *
sigrange=3/2

def innerproductBSsx(bra,ket): #bs= basic source 
    out=0
    for ssbra in range(bra.nofss):
        gbeambra=bra.subsources[ssbra]
        for ssket in range(ket.nofss):
            out+=innerproductGBsx(bra=gbeambra,ket=ket.subsources[ssket])
    return out
class qstate:
    def __init__(self):
        self.setxyranges()
    def sourceatz(self,z):
        return self
    def intensityxatz(self,z,xs):
        self.goto(z)
        return self.intensityx(xs)
    def intensitykxatz(self,z,kxs):
        self.goto(z)
        return self.intensitykx(kxs)
    def plotintensityxatz(self,z,nofxbins=300,xs=np.array([-1]),rescaleamp=1,paritytransform=False,symmetrize=False,onlygivedata=False,xunit='cm'):
        if xs.size<3:
            minx=np.min(xs) if xs.size==2 else self.minx
            maxx=np.max(xs) if xs.size==2 else self.maxx
            if symmetrize:
                maxx=max(abs(minx),abs(maxx))
                minx=-maxx
            xs=np.linspace(minx,maxx,nofxbins)

        rescalesign= -1 if paritytransform else 1  
        amps=rescaleamp*self.intensityxatz(z,xs*abs(rescaleamp)**2*rescalesign) # it is for FT and convolution test
        if not onlygivedata:
            fig = plt.figure() 
            ax = fig.add_subplot(1, 1, 1)
            fig.suptitle(' intensity x at z = '+str(z)+" "+xunit, fontsize=16)
            phasedplotv2(ax,xs,amps,kind='x',xunit=xunit) 
        return xs, amps
class BSource(qstate): #basic source 
    ampll =10**(-4) # amplitude lower limit;  note that amp^2 gives the probability  
    res=10**(-6) # resolution to do discretization 
    def __init__(self,nofss,K,ftype='fockstate',fvalues=1,z0s=[],sigxs=np.array([]),sigkxs=np.array([]),sigys=np.array([]),sigkys=np.array([]),x0s=np.array([]),y0s=np.array([]),kx0stheta=np.array([]),ky0stheta=np.array([]),kx0s=np.array([]),ky0s=np.array([]),amps=np.array([]),nofsigranges=np.array([]),longphase=False):
        # Note: DO NOT USE beam relate info from source, such as source.amps, source.x0s, ....
        
        self.nofss=nofss # number of sub sources
        self.k=K
        self.longphase=longphase  #longphase=longitudinal phase
        self.ftype=ftype #ftype= f function of quantum source
        self.fvalues=fvalues
        self.z0s=np.full((nofss), 0.0) if len(z0s)==0 else np.array(z0s)
        self.sigxs=np.full((nofss), -1.0) if len(sigxs)==0 else np.array(sigxs)       
        self.sigkxs=np.full((nofss), -1.0) if len(sigkxs)==0 else np.array(sigkxs)
        self.sigys=np.full((nofss), -1.0) if len(sigys)==0 else np.array(sigys)
        self.sigkys=np.full((nofss), -1.0) if len(sigkys)==0 else np.array(sigkys)
        self.x0s=np.full((nofss), 0.0) if len(x0s)==0 else np.array(x0s)
        self.y0s=np.full((nofss), 0.0) if len(y0s)==0 else np.array(y0s)

        if len(kx0stheta)!=0 and len(kx0s)!=0:
            err=1/0
        elif len(kx0s)!=0:
             self.kx0s=np.array(kx0s)
        elif len(kx0stheta)!=0:
            self.kx0s= self.k*np.sin(kx0stheta)
        else:
            self.kx0s=np.full((nofss), 0.0)       
        if len(ky0stheta)!=0 and len(ky0s)!=0:
            err=1/0
        elif len(ky0s)!=0:
             self.ky0s=np.array(ky0s)
        elif len(ky0stheta)!=0:
            self.ky0s= self.k*np.sin(ky0stheta)
        else:
            self.ky0s=np.full((nofss), 0.0)

        self.amps=np.full((nofss), 1) if len(amps)==0 else np.array(amps)
        self.nofsigranges=np.full((nofss), sigrange) if len(nofsigranges)==0 else np.array(nofsigranges)
    
        self.subsources=np.empty([nofss],dtype=GBeam)
        for ss in range(nofss): 
            self.subsources[ss]=GBeam(K=self.k,z0=self.z0s[ss],
            sigx=self.sigxs[ss],sigkx=self.sigkxs[ss],
            sigy=self.sigys[ss],sigky=self.sigkys[ss],
            x0=self.x0s[ss],y0=self.y0s[ss],
            kx0=self.kx0s[ss],ky0=self.ky0s[ss],
            amp=self.amps[ss],nofsigrange=self.nofsigranges[ss]
            ,longphase=longphase)
        qstate.__init__(self)

    def addbeamviaparameter(self,z0=0,sigx=-1,sigkx=-1,sigy=-1,sigky=-1,x0=0,y0=0,kx0=0,ky0=0,amp=1,nofsigrange=3/2):
        self.nofss+=1 # number of sub sources
        self.z0s=np.append(self.z0s,z0)
        self.sigxs=np.append(self.sigxs,sigx)       
        self.sigkxs=np.append(self.sigkxs,sigkx)
        self.sigys=np.append(self.sigys,sigy)
        self.sigkys=np.append(self.sigkys,sigky)
        self.x0s=np.append(self.x0s,x0)
        self.y0s=np.append(self.y0s,y0)
        self.kx0s=np.append(self.kx0s,kx0) 
        self.ky0s=np.append(self.ky0s,ky0)
        self.amps=np.append(self.amps,amp)
        self.nofsigranges=np.append(self.nofsigranges,nofsigrange)
    
        self.subsources=np.append(self.subsources,GBeam(K=self.k,z0=z0,
            sigx=sigx,sigkx=sigkx,
            sigy=sigy,sigky=sigky,
            x0=x0,y0=y0,
            kx0=kx0,ky0=ky0,
            amp=amp,nofsigrange=nofsigrange))
    def addbeam(self,beam):
        self.nofss+=1 # number of sub sources
        self.z0s=np.append(self.z0s,beam.z)
        self.sigxs=np.append(self.sigxs,beam.sigxatz)       
        self.sigkxs=np.append(self.sigkxs,beam.sigkxatz)
        self.sigys=np.append(self.sigys,beam.sigyatz)
        self.sigkys=np.append(self.sigkys,beam.sigkyatz)
        self.x0s=np.append(self.x0s,beam.x0atz)
        self.y0s=np.append(self.y0s,beam.y0atz)
        self.kx0s=np.append(self.kx0s,beam.kx0) 
        self.ky0s=np.append(self.ky0s,beam.ky0) 
        self.amps=np.append(self.amps,beam.ampatz)
        self.nofsigranges=np.append(self.nofsigranges,beam.nofsigrange)
    
        self.subsources=np.append(self.subsources,GBeam())
        self.subsources[self.nofss-1]=copy.deepcopy(beam)

    def setminwaistsx(self, minwaists,dztominwaists=np.array([])):
        dztomws=np.full((self.nofss), 0.0) if dztominwaists.size==0 else dztominwaists
        for ss in range(self.nofss):
            self.subsources[ss].setminwaistx(minwaists[ss],dztomws[ss])
    def setminwaistsy(self, minwaists,dztominwaists=np.array([])):
        dztomws=np.full((self.nofss), 0.0) if dztominwaists.size==0 else dztominwaists
        for ss in range(self.nofss):
            self.subsources[ss].setminwaisty(minwaists[ss],dztomws[ss])
    
    def setxyranges(self):
        self.subsources[0].setxyranges()

        self.minx=self.subsources[0].minx
        self.maxx=self.subsources[0].maxx
        self.minkx=self.subsources[0].minkx
        self.maxkx=self.subsources[0].maxkx

        self.miny=self.subsources[0].miny
        self.maxy=self.subsources[0].maxy
        self.minky=self.subsources[0].minky
        self.maxky=self.subsources[0].maxky

        for ss in self.subsources:
            ss.setxyranges()
            self.minx=min(self.minx,ss.minx)
            self.maxx=max(self.maxx,ss.maxx)
            self.minkx=min(self.minkx,ss.minkx)
            self.maxkx=max(self.maxkx,ss.maxkx)

            self.miny=min(self.miny,ss.miny)
            self.maxy=max(self.maxy,ss.maxy)
            self.minky=min(self.minky,ss.minky)
            self.maxky=max(self.maxky,ss.maxky)

    def setranges(self, zmin, zmax):
        self.zmin=zmin
        self.zmax=zmax
        self.subsources[0].setranges(zmin, zmax)

        self.minx=self.subsources[0].minx
        self.maxx=self.subsources[0].maxx
        self.minkx=self.subsources[0].minkx
        self.maxkx=self.subsources[0].maxkx

        self.miny=self.subsources[0].miny
        self.maxy=self.subsources[0].maxy
        self.minky=self.subsources[0].minky
        self.maxky=self.subsources[0].maxky

        for ss in self.subsources:
            ss.setranges(zmin, zmax)
            self.minx=min(self.minx,ss.minx)
            self.maxx=max(self.maxx,ss.maxx)
            self.minkx=min(self.minkx,ss.minkx)
            self.maxkx=max(self.maxkx,ss.maxkx)

            self.miny=min(self.miny,ss.miny)
            self.maxy=max(self.maxy,ss.maxy)
            self.minky=min(self.minky,ss.minky)
            self.maxky=max(self.maxky,ss.maxky)
        

    def goto(self, z):
        for ss in range(self.nofss):
            self.subsources[ss].goto(z)
    def passlens(self, lens):
        for ss in range(self.nofss):
            self.subsources[ss].passlens(lens)
    def passslmpix(self,slmpix):
         for ss in range(self.nofss):
            self.subsources[ss].passslmpix(slmpix)
    def getampsatz(self,z):
        self.goto(z)
        amps=np.array([],dtype=complex)
        x0s=np.array([],dtype=float)
        y0s=np.array([],dtype=float)
        kx0s=np.array([],dtype=float)
        ky0s=np.array([],dtype=float)
        sigxs=np.array([],dtype=complex)
        sigys=np.array([],dtype=complex)
        for ss in range(self.nofss):
            x0ss=self.subsources[ss].x0atz
            y0ss=self.subsources[ss].y0atz
            kx0ss=self.subsources[ss].kx0
            ky0ss=self.subsources[ss].ky0
            ampss=self.subsources[ss].ampatz
            sigxss=self.subsources[ss].sigxatz
            sigyss=self.subsources[ss].sigyatz
            y0ss=self.subsources[ss].y0atz
            ampinds=np.where(abs(x0s-x0ss) + abs(y0s-y0ss)+ abs(kx0s-kx0ss)+ abs(ky0s-ky0ss)+ abs(sigxs-sigxss)+ abs(sigys-sigyss) <self.res)[0]
            if len(ampinds)==0:
                amps=np.append(amps,ampss)
                x0s=np.append(x0s,x0ss)
                y0s=np.append(y0s,y0ss)
                kx0s=np.append(kx0s,kx0ss)
                ky0s=np.append(ky0s,ky0ss)
                sigxs=np.append(sigxs,sigxss)
                sigys=np.append(sigys,sigyss)
            elif len(ampinds)==1:
                amps[ampinds[0]] += ampss
            else:
                print('increase resolution res!')
                sys.exit('increase resolution res!')
        
        outamps=np.array([],dtype=complex)
        outx0s=np.array([],dtype=float)
        outy0s=np.array([],dtype=float)
        outkx0s=np.array([],dtype=float)
        outky0s=np.array([],dtype=float)
        outsigxs=np.array([],dtype=complex)
        outsigys=np.array([],dtype=complex)
        for ind in range(amps.size):
            if abs(amps[ind]) > self.ampll:
                outamps=np.append(outamps,amps[ind])
                outx0s=np.append(outx0s,x0s[ind])
                outy0s=np.append(outy0s,y0s[ind])
                outkx0s=np.append(outkx0s,kx0s[ind])
                outky0s=np.append(outky0s,ky0s[ind])
                outsigxs=np.append(outsigxs,sigxs[ind])
                outsigys=np.append(outsigys,sigys[ind])
        return outx0s,outy0s,outamps,outkx0s,outky0s,outsigxs,outsigys

    def passslm00(self, slm):
        prenofss=self.nofss
        for slmind in range(slm.size-1):
            for ss in range(prenofss):
                self.addbeam(self.subsources[ss])
        ind=0
        for slmind in range(slm.size):
            amp=slm.amps[slmind]
            dkx= slm.ks[slmind] if slm.ks.ndim==1 else slm.ks[0][slmind]
            dky= 0 if slm.ks.ndim==1 else slm.ks[1][slmind]
            for ss in range(prenofss):
                self.subsources[ind].rotate(amp=amp,dkx=dkx,dky=dky)
                ind+=1
    def passslm(self, slm):
        prenofss=self.nofss
        for slmind in range(slm.size-1):
            for ss in range(prenofss):
                self.addbeam(self.subsources[ss])
        ind=0
        for slmind in range(slm.size):
            amp=slm.amps[slmind]
            dkx= slm.ks[slmind] if slm.ks.ndim==1 else slm.ks[0][slmind]
            dky= 0 if slm.ks.ndim==1 else slm.ks[1][slmind]
            for ss in range(prenofss):
                self.subsources[ind].rotate(amp=amp,dkx=dkx,dky=dky)
                ind+=1
    def wpx(self,xs):
        outamps=np.zeros_like(xs,dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpx(xs)
        return outamps
    def wpy(self,ys):
         outamps=np.zeros_like(ys,dtype=complex)
         for ss in range(self.nofss):
             outamps+=self.subsources[ss].wpy(ys)
         return outamps    
    def wpky(self,kys):
        outamps=np.zeros_like(kys,dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpky(kys)
        return outamps
    def wpkx(self,kxs):
        outamps=np.zeros_like(kxs,dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpkx(kxs)
        return outamps

    def wpxy(self,xs,ys):
        outamps=np.zeros((xs.size,ys.size),dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpxy(xs,ys)
        return outamps
    def wpxylinear(self,xs,ys):
        outamps=np.zeros_like(xs,dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpxylinear(xs,ys)
        return outamps

    def wpkxy(self,kxs,kys):
        outamps=np.zeros((kxs.size,kys.size),dtype=complex)
        for ss in range(self.nofss):
            outamps+=self.subsources[ss].wpkxy(kxs,kys)
        return outamps
    
    def wpxatz(self,z,xs):
        self.goto(z)
        return self.wpx(xs)
    def wpyatz(self,z,ys):
        self.goto(z)
        return self.wpy(ys)
    def wpkxatz(self,z,kxs):
        self.goto(z)
        return self.wpkx(kxs)
    def wpkyatz(self,z,kys):
        self.goto(z)
        return self.wpky(kys)
    def wpxyatz(self,z,xs,ys):
        self.goto(z)
        return self.wpxy(xs,ys)
    def wpkxyatz(self,z,kxs,kys):
        self.goto(z)
        return self.wpkxy(kxs,kys)
        
    def wpxylinearatz(self,z,xs,ys):
        self.goto(z)
        return self.wpxylinear(xs,ys)
    def intensityx(self,xs):
        return abs(self.wpx(xs))**2
    def intensitykx(self,kxs):
        return abs(self.wpkx(kxs))**2