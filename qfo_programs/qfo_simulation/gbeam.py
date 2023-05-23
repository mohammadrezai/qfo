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

import numpy as np
import cmath
import sys
sigrange=3/2
def innerproductGBsx(bra,ket): # inner product of two Gaussian Beam
    if ket.sigyatz!=bra.sigyatz or ket.y0atz!=bra.y0atz:
        print("ERROR: No y overlap \n innerproductGBsx function is for y overlapping beams ")
        sys.exit()

    if ket.sigxatz==bra.sigxatz:
        ampexp=1
    else:
        ampexp=(ket.sigxatz.real/(ket.sigxatz*abs(ket.sigxatz)))**(1/4)
        ampexp*=np.conj((bra.sigxatz.real/(bra.sigxatz*abs(bra.sigxatz)))**(1/4))
        ampexp/=(1/ket.sigxatz + np.conj(1/bra.sigxatz))**(1/2)
        ampexp*=2**(1/2)
    
    ampexp*=ket.ampatz*np.conj(bra.ampatz)*np.exp((-(ket.x0atz-bra.x0atz)**2/(ket.sigxatz+np.conj(bra.sigxatz)) - 1j*(ket.kx0*ket.x0atz-bra.kx0*bra.x0atz))/2)
    return ampexp
class GBeam: # Gaussian Beam
    def __init__(self,K=600,z0=0,sigx=-1,sigkx=-1,sigy=-1,sigky=-1,x0=0,y0=0,kx0=0,ky0=0,amp=1,nofsigrange=sigrange,longphase=False):
        self.ampatz = amp
        self.z=z0
        self.x0atz=x0
        self.y0atz=y0
        self.kx0=kx0
        self.ky0=ky0
        self.K=K
        self.longphase=longphase
        if sigx !=-1 and sigkx==-1:
            self.sigxatz=sigx
            self.sigkxatz=1/sigx
        elif sigkx!=-1 and sigx==-1:
            self.sigkxatz=sigkx
            self.sigxatz=1/sigkx
        elif sigkx==-1 and sigx==-1:
            self.sigxatz=1
            self.sigkxatz=1
        else:
            err=1/0

        if sigy !=-1 and sigky==-1:
            self.sigyatz=sigy
            self.sigkyatz=1/sigy
        elif sigky!=-1 and sigy==-1:
            self.sigkyatz=sigky
            self.sigyatz=1/sigky
        elif sigky==-1 and sigy==-1:
            self.sigyatz=1
            self.sigkyatz=1
        else:
            err=1/0
        
        self.nofsigrange=nofsigrange
        self.setzasz0()

    def getminmax(self,center,complexsig):
        invsig=1/complexsig
        range=self.nofsigrange*1/abs(invsig.real)**(1/2)
        return center-range, center+range
    def getmima(self,coordinate='x'):
        if coordinate=='x':
            return self.getminmax(center=self.x0atz,complexsig=self.sigxatz)
        elif coordinate=='y':
            return self.getminmax(center=self.y0atz,complexsig=self.sigyatz)
        elif coordinate=='kx':
            return self.getminmax(center=self.kx0,complexsig=self.sigkxatz)
        elif coordinate=='ky':
            return self.getminmax(center=self.ky0,complexsig=self.sigkyatz)

    def checkminmax(self,center,complexsig,min0,max0):
        invsig=1/complexsig
        range=self.nofsigrange*1/abs(invsig.real)**(1/2)
        return min(min0,center-range), max(max0,center+range)

    def setxyranges(self):
        self.minx, self.maxx=self.getminmax(center=self.x0atz,complexsig=self.sigxatz)
        self.miny, self.maxy=self.getminmax(center=self.y0atz,complexsig=self.sigyatz)
        self.minkx, self.maxkx=self.getminmax(center=self.kx0,complexsig=self.sigkxatz)
        self.minky, self.maxky=self.getminmax(center=self.ky0,complexsig=self.sigkyatz)
    
    def checkxyranges(self):
        self.minx, self.maxx=self.checkminmax(center=self.x0atz,complexsig=self.sigxatz,min0=self.minx,max0=self.maxx)
        self.miny, self.maxy=self.checkminmax(center=self.y0atz,complexsig=self.sigyatz,min0=self.miny,max0=self.maxy)
        self.minkx, self.maxkx=self.checkminmax(center=self.kx0,complexsig=self.sigkxatz,min0=self.minkx,max0=self.maxkx)
        self.minky, self.maxky=self.checkminmax(center=self.ky0,complexsig=self.sigkyatz,min0=self.minky,max0=self.maxky)
    def setranges(self, zmin, zmax):
        self.zmin=zmin
        self.zmax=zmax
        self.goto(zmin)
        self.setxyranges()
        self.goto(zmax)
        self.checkxyranges()
    def setzasz0(self):
        self.z0=self.z
        self.ampatz0=self.ampatz
        self.x0atz0=self.x0atz
        self.y0atz0=self.y0atz
        self.sigxatz0=self.sigxatz
        self.sigyatz0=self.sigyatz
        self.sigkxatz0=self.sigkxatz
        self.sigkyatz0=self.sigkyatz

    def setminwaistx(self,minwaist,dztominwaist=0):
        imsigx=-dztominwaist/self.K
        self.sigxatz=2*(minwaist/6)**2 +1j*imsigx
        self.sigkxatz=1/self.sigxatz
        self.setzasz0()

    def setminwaisty(self,minwaist,dztominwaist=0):
        imsigy=-dztominwaist/self.K
        self.sigyatz=2*(minwaist/6)**2 +1j*imsigy
        self.sigkyatz=1/self.sigyatz
        self.setzasz0()

    def goto(self,z):
        self.z=z
        dz=z-self.z0
        self.x0atz=self.x0atz0+(self.kx0*dz/self.K)
        self.sigkxatz=(self.K *self.sigkxatz0)/(self.K+1j*dz*self.sigkxatz0)
        self.sigxatz=1/self.sigkxatz

        self.y0atz=self.y0atz0+(self.ky0*dz/self.K)
        self.sigkyatz=(self.K *self.sigkyatz0)/(self.K+1j*dz*self.sigkyatz0)
        self.sigyatz=1/self.sigkyatz
        self.ampatz=self.ampatz0*cmath.exp(-1j*(cmath.phase(self.K+1j*dz*self.sigkxatz0)/4+cmath.phase(self.K+1j*dz*self.sigkyatz0)/4))
        if (self.longphase): #longphase =longitudinal phase
            self.ampatz*=cmath.exp(1j*(self.K*dz))
    def passlens(self, lens):
        lenfac=-self.K/(2*lens.f)
        self.ampatz*=cmath.exp(-1j*(cmath.phase(1-2j*lenfac*self.sigxatz)/4))  
        self.ampatz*=cmath.exp(-1j*(cmath.phase(1-2j*lenfac*self.sigyatz)/4))  
       
        self.kx0=self.kx0 + 2*lenfac*self.x0atz
        self.ky0=self.ky0 + 2*lenfac*self.y0atz

        self.sigxatz=1/(1/self.sigxatz-2j*lenfac)
        self.sigyatz=1/(1/self.sigyatz-2j*lenfac)
        self.sigkxatz=1/self.sigxatz
        self.sigkyatz=1/self.sigyatz
        self.setzasz0()

    def passslmpix(self,slmpix):
        if slmpix.dim==1:
            ampinds=np.where(abs(slmpix.pixsx-self.x0atz) < (slmpix.pixsx[1]-slmpix.pixsx[0])/10) 
        elif slmpix.dim==2:
            ampinds=np.where(np.logical_and(slmpix.pixsx == self.x0atz, slmpix.pixsy ==self.x0atz))[0]
        if len(ampinds)==1:
            self.ampatz*=slmpix.amps[ampinds[0]]
        else:
            print('no match at slmpix')
            sys.exit('no match at slmpix')
        self.setzasz0()



    def rotate(self, amp=1,dkx=0,dky=0):
        self.kx0+=dkx
        self.ky0+=dky 
        self.ampatz*=amp*cmath.exp(1j*(self.x0atz*dkx+self.y0atz*dky)/2)
        self.setzasz0()
    def wpx(self,xs):        
    #    Gaussian wave packet at x
        ampexp=self.ampatz*(self.sigxatz.real/(np.pi* self.sigxatz*abs(self.sigxatz)))**(1/4)
        return ampexp*np.exp(-(xs-self.x0atz)**2/(2*self.sigxatz) +1j *xs * self.kx0- 1j *self.kx0*self.x0atz/2)
    def wpy(self,ys):        
    #    Gaussian wave packet at x
        ampexp=self.ampatz*(self.sigyatz.real/(np.pi* self.sigyatz*abs(self.sigyatz)))**(1/4)
        return ampexp*np.exp(-(ys-self.y0atz)**2/(2*self.sigyatz) +1j *ys * self.ky0- 1j *self.ky0*self.y0atz/2)

    def wpxy(self,xs,ys):
    #    Gaussian wave packet at x
        ampexp=self.ampatz*(self.sigxatz.real/(np.pi* self.sigxatz*abs(self.sigxatz)))**(1/4)
        ampexp*=           (self.sigyatz.real/(np.pi* self.sigyatz*abs(self.sigyatz)))**(1/4)
        ampexp*=np.exp(-1j *self.kx0*self.x0atz/2)
        ampexp*=np.exp(-1j *self.ky0*self.y0atz/2)
        out=np.empty([xs.size,ys.size],dtype=complex)
        for ix in range(xs.size): 
            out[ix,:]=ampexp*np.exp(-(xs[ix]-self.x0atz)**2/(2*self.sigxatz) +1j *xs[ix] * self.kx0)*np.exp(-(ys-    self.y0atz)**2/(2*self.sigyatz) +1j *ys * self.ky0)
        return out

    def wpxylinear(self,xs,ys):
    #    Gaussian wave packet at x
        ampexp=self.ampatz*(self.sigxatz.real/(np.pi* self.sigxatz*abs(self.sigxatz)))**(1/4)
        ampexp*=           (self.sigyatz.real/(np.pi* self.sigyatz*abs(self.sigyatz)))**(1/4)
        ampexp*=np.exp(-1j *self.kx0*self.x0atz/2)
        ampexp*=np.exp(-1j *self.ky0*self.y0atz/2)
        out=np.empty([xs.size],dtype=complex)
        out=ampexp*np.exp(-(xs-self.x0atz)**2/(2*self.sigxatz) +1j *xs * self.kx0)*np.exp(-(ys-    self.y0atz)**2/(2*self.sigyatz) +1j *ys * self.ky0)
        return out
            
    def wpkxy(self,kxs,kys):
    #    Gaussian wave packet at kx
        ampexp=self.ampatz*(self.sigkxatz.real/(np.pi* self.sigkxatz*abs(self.sigkxatz)))**(1/4)
        ampexp*=           (self.sigkyatz.real/(np.pi* self.sigkyatz*abs(self.sigkyatz)))**(1/4)
        ampexp*=np.exp(1j *self.kx0*self.x0atz/2)
        ampexp*=np.exp(1j *self.ky0*self.y0atz/2)
        out=np.empty([kxs.size,kys.size],dtype=complex)
        for ix in range(kxs.size): 
            out[ix,:]=ampexp*np.exp(-(kxs[ix]-self.kx0)**2/(2*self.sigkxatz) -1j *kxs[ix] * self.x0atz)* np.exp(-(kys    -self.ky0)**2/(2*self.sigkyatz) -1j *kys * self.y0atz)
        return out
    
    def wpkx(self,kxs):
    #    Gaussian wave packet at kx
        amp=self.ampatz*(self.sigkxatz.real/(np.pi* self.sigkxatz*abs(self.sigkxatz)))**(1/4)
        return amp*np.exp(-(kxs-self.kx0)**2/(2*self.sigkxatz) -1j *kxs * self.x0atz + 1j *self.kx0*self.x0atz/2)
    def wpky(self,kys):
    #    Gaussian wave packet at kx
        amp=self.ampatz*(self.sigkyatz.real/(np.pi* self.sigkyatz*abs(self.sigkyatz)))**(1/4)
        return amp*np.exp(-(kys-self.ky0)**2/(2*self.sigkyatz) -1j *kys * self.y0atz + 1j *self.ky0*self.y0atz/2)

    def hfr(self,dz):
    #    Gaussian wave packet at r
        return cmath.exp(1j *self.K*dz*(1-(self.kx0**2)/(2*self.K**2)-(self.ky0**2)/(2*self.K**2)))