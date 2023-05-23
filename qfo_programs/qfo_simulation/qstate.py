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

from source import *
from itertools import permutations
np.set_printoptions(suppress=True)


def innerproductFSsx(bra,ket): 
    BSpwprods=giveinnprodBSofFS(bra_bsources=bra,ket_bsources=ket)#BSpwprod=pairwise products of Basic Sources
    return giveinnprodFSsx(BSpwprods)

def giveinnprodFSsx(BSpwprods): #BSpwprod=pairwise products of Basic Sources
    branofs=BSpwprods.shape[0]
    ketnofs=BSpwprods.shape[1]
    out=0
    if branofs==ketnofs:
        perm = permutations(range(branofs))
        for myperm in perm:
            myperminnprod=1
            for ind in range(ketnofs):
                braind=myperm[ind]
                myperminnprod*=BSpwprods[braind,ind]
            out+=myperminnprod
    return out 

def giveinnprodBSofFS(bra_bsources,ket_bsources): #pairwise products of Basic Sources from Factorized sources
    """inner product of Basic sources of Factrorized sources"""
    out=np.empty([len(bra_bsources),len(ket_bsources)],dtype=complex)
    for indsbra in range(len(bra_bsources)):
        sbra=bra_bsources[indsbra]
        for indsket in range(len(ket_bsources)):
            out[indsbra,indsket]=innerproductBSsx(bra=sbra,ket=ket_bsources[indsket])
    return out


class Dn:  # discrete fock (number) state 
    def __init__(self,k,n=1,x0=0,y0=0,kx0=0,ky0=0,sigx=1,sigy=1):
        self.k=k
        self.n=n
        self.x0=x0
        self.y0=y0
        self.kx0=kx0
        self.ky0=ky0
        self.sigx=sigx
        self.sigy=sigy
    def isequal(self, ns,res):
        diff=abs(ns.k-self.k)+abs(ns.x0-self.x0)+abs(ns.y0-self.y0)+abs(ns.kx0-self.kx0)+abs(ns.ky0-self.ky0)+abs(ns.sigx-self.sigx)+abs(ns.sigy-self.sigy)
        if self.n==ns.n and diff<res:
            return True
        else:
            return False

class FDn: # factrorized of discrete number states: amp|n1,n2,n3,...>
    res=BSource.res
    def __init__(self,amp=1+0j,dx=1,dy=1):
        self.Dns=np.array([],dtype=Dn)
        self.dx=dx#dx = distance between discretized wavefront: i=x/dx for |n_{i,j}>
        self.dy=dy#dy = distance between discretized wavefront: j=y/dy for |n_{i,j}>
        self.amp=amp
        self.totn=0
    def removezeros(self):
        preDns=self.Dns
        self.Dns=np.array([],dtype=Dn)
        self.totn=0
        for dn in preDns:
            if dn.n!=0:
                self.Dns=np.append(self.Dns,dn)
                self.totn+=dn.n
    def addedifequal (self,fdn): #|n_2, m_3> = |m_3,n_2>
        indfdn=0
        isequal=True
        if self.totn!= fdn.totn:
            print('uneq totns:',self.totn, fdn.totn)
            test1=[dn.n for dn in self.Dns]
            test2=[dn.n for dn in fdn.Dns]
            print(test1,test2)
            err=1/0
        if fdn.Dns.size!=self.Dns.size:
            return False
        while isequal and indfdn<fdn.Dns.size:
            isequal=False
            indself=0
            while not isequal and indself<self.Dns.size:
                isequal = fdn.Dns[indfdn].isequal(self.Dns[indself],self.res)
                indself+=1
            indfdn+=1
        if isequal and indfdn==fdn.Dns.size:
            self.amp+=fdn.amp
            return True
        else:
            return False
    def applyad(self,k,x0,y0,kx0=0,ky0=0,sigx=1,sigy=1,amp=1,nofad=1): #applyad ad^nofad
        newmode=True
        for dn in self.Dns:
            diff=abs(dn.k-k)+abs(dn.x0-x0)+abs(dn.y0-y0)+abs(dn.kx0-kx0)+abs(dn.ky0-ky0)+abs(dn.sigx-sigx)+abs(dn.sigy-sigy)
            if diff<self.res:
                newmode=False
                self.amp*=amp
                for iin in range(nofad):
                    self.totn+=1
                    dn.n+=1
                    self.amp*=(dn.n)**(1/2)
        if newmode:
            newDn=Dn(k=k,n=nofad,x0=x0,y0=y0,kx0=kx0,ky0=ky0,sigx=sigx,sigy=sigy)
            self.Dns=np.append(self.Dns,newDn)
            self.amp*=amp*(np.math.factorial(nofad))**(1/2)
            self.totn+=nofad
            
    def givefsourcewoamp(self,z0,nofsigranges,longphase): # wo without amp; amp=P_j 1/sqrt(n_j!)
        nbsources=[dn.n for dn in self.Dns]
        bsources=[BSource(nofss=1,K=dn.k,z0s=[z0],sigxs=[dn.sigx],sigys=[dn.sigy],x0s=[dn.x0],y0s=[dn.y0],kx0s=[dn.kx0],ky0s=[dn.ky0],amps=[1],nofsigranges=nofsigranges,longphase=longphase) for dn in self.Dns]
        return FSource(bsources=bsources,nbsources=nbsources,dx=self.dx,dy=self.dy,bsourcesold=[],activeold=False)
    
    def givefsourceamp(self): # wo without amp; amp=P_j 1/sqrt(n_j!)
        famp=self.amp
        for dn in self.Dns:
            famp*=1/(np.math.factorial(dn.n))**(1/2)
        return famp
    def printbase(self,rounddig=4,showy=True,absamps=False,ampmultiplier=1):
        if absamps:
            amp=round(abs(self.amp*ampmultiplier),rounddig)
        else:
            amp=round(self.amp*ampmultiplier,rounddig)
        if (amp!=0):
            print('+',amp,"|", end = '')
            for dn in self.Dns:
                if showy:
                    print("(",dn.n,')_{',round(dn.x0/self.dx),',',round(dn.y0/self.dy),'}', end = '')
                else:
                    print("(",dn.n,')_{',round(dn.x0/self.dx),'}', end = '')
            print(">", end = '')
    
    def project(self,projns=[],projxs=[],projys=[],demolish=False):
        if (self.amp!=0):
            if len(projxs)==len(projys)==len(projns):
                projectsize=len(projxs)
            else:
                1/0
            iproj=0
            projected=False
            iprojected=projectsize
            while not projected and iproj <projectsize:
                myxs=projxs[iproj]
                myys=projys[iproj]
                myns=projns[iproj]
                if len(myxs)==len(myys)==len(myns):
                    mylen=len(myns)
                else:
                    1/0
                myprojected=True
                myii=0
                while myii<mylen and myprojected==True:
                    myprojected=False
                    myx=myxs[myii]
                    myy=myys[myii]
                    myn=myns[myii]
                    for ns in self.Dns:
                        if myx==round(ns.x0/self.dx) and myy==round(ns.y0/self.dy) and myn==ns.n:
                            myprojected=True
                    myii+=1
                if myprojected==True and myii==mylen:
                    projected=True
                    iprojected=iproj
                iproj+=1
            while iproj <projectsize:
                myxs=projxs[iproj]
                myys=projys[iproj]
                myns=projns[iproj]
                if len(myxs)==len(myys)==len(myns):
                    mylen=len(myns)
                else:
                    1/0
                myprojected=True
                myii=0
                while myii<mylen and myprojected==True:
                    myprojected=False
                    myx=myxs[myii]
                    myy=myys[myii]
                    myn=myns[myii]
                    for ns in self.Dns:
                        if myx==round(ns.x0/self.dx) and myy==round(ns.y0/self.dy) and myn==ns.n:
                            myprojected=True
                    myii+=1
                if myprojected==True and myii==mylen:
                    print('projected on two cases!')
                    err=1/0
                iproj+=1
            if projected and demolish:
                iproj=iprojected
                myxs=projxs[iproj]
                myys=projys[iproj]
                myns=projns[iproj]
                if len(myxs)==len(myys)==len(myns):
                    mylen=len(myns)
                else:
                    1/0
                myprojected=True
                myii=0
                while myii<mylen and myprojected==True:
                    myprojected=False
                    myx=myxs[myii]
                    myy=myys[myii]
                    myn=myns[myii]
                    for ns in self.Dns:
                        if myx==round(ns.x0/self.dx) and myy==round(ns.y0/self.dy) and myn==ns.n:
                            if ns.n==1:
                                ns.n=0
                            else:
                                err=1/0  # for more than one photon demolishing we need to apply a on the number state
                                         # a|n> = sqrt(n)|n-1>  or a^n |n> =sqrt(n!)|0>?
                                         # amp*= sqrt(n)?  or amp*=sqrt(n!)
                            myprojected=True
                    myii+=1
                if myprojected==True and myii==mylen:
                    pass
                else:
                    err=1/0
            return projected
        else:
            err=1/0
    def printbaseifprojected(self,rounddig=4,showy=True,absamps=False,projns=[],projxs=[],projys=[],ampmultiplier=1,demolish=False):
        projected=self.project(projns=projns,projxs=projxs,projys=projys,demolish=demolish)
        if absamps:
            amp=round(abs(self.amp*ampmultiplier),rounddig)
        else:
            amp=round(self.amp*ampmultiplier,rounddig)
        if projected and amp!=0:
            print('+',amp,"|", end = '')
            for ns in self.Dns:
                if showy:
                    print("(",ns.n,')_{',round(ns.x0/self.dx),',',round(ns.y0/self.dy),'}', end = '')
                else:
                    print("(",ns.n,')_{',round(ns.x0/self.dx),'}', end = '')
            print(">", end = '')
        return projected
class ESource(qstate): # Entangled version
    def __init__(self,fsources,amps,ExpandSimplify=False):
        # Note: DO NOT USE beam related info from source, such as source.amps, source.x0s, ....
        self.longphase=fsources[0].longphase  #longphase=longitudinal phase
        self.fsources=[copy.deepcopy(fs) for fs in fsources]
        if len(amps)==0:
            self.amps=np.array([1  for i in range(len(fsources))],dtype=complex)

        elif len(amps)==len(fsources):
            self.amps=np.array(amps,dtype=complex)
        else:
            err=1/0
        if ExpandSimplify:
            z=fsources[0].bsources[0].subsources[0].z
            self.ExpandSimplifyatz(z)
        else:
            self.zFDnarr=-11231414245.123131
        qstate.__init__(self)  
    def project(self,z,projns,projxs,projys,demolish=False):
        for fsource in self.fsources:
            fsource.projectFDnarr(z=z,projns=projns,projxs=projxs,projys=projys,demolish=demolish)
        ifs=0
        while ifs<len(self.fsources) and self.fsources[ifs].FDnarr.size==0:
            ifs+=1
        if ifs<len(self.fsources):
            fsource=self.fsources[ifs]   
            ampfs=self.amps[ifs]  
            self.FDnarr=fsource.FDnarr
            for fdn in self.FDnarr:
                fdn.amp*=ampfs

            self.zFDnarr=z
            ifs+=1
            while ifs < len(self.fsources):
                fsource=self.fsources[ifs]   
                ampfs=self.amps[ifs] 
                fdnarr_fs=fsource.FDnarr
                for fdn_fs in fdnarr_fs:
                    indFDnarr=0
                    fdn_fs.amp*=ampfs
                    while indFDnarr<self.FDnarr.size and not self.FDnarr[indFDnarr].addedifequal(fdn_fs):
                        indFDnarr+=1
                    if indFDnarr==self.FDnarr.size:
                        self.FDnarr=np.append(self.FDnarr,fdn_fs)
                ifs+=1
        self.fsources=[fdn.givefsourcewoamp(z0=z,nofsigranges=np.array([]),longphase=fsource.longphase) for fdn in self.FDnarr]
        self.amps=np.array([fdn.givefsourceamp() for fdn in self.FDnarr])



    def ExpandSimplifyatz(self,z): 
        for fsource in self.fsources:
            fsource.setfockrepatz(z=z)
        fsource=self.fsources[0]    
        self.FDnarr=fsource.FDnarr
        for fdn in self.FDnarr:
            fdn.amp*=self.amps[0]
        self.amps[0]=1
        self.zFDnarr=z
        for ifs in range(1,len(self.fsources)):
            fsource=self.fsources[ifs]   
            ampfs=self.amps[ifs] 
            fdnarr_fs=fsource.FDnarr
            for fdn_fs in fdnarr_fs:
                indFDnarr=0
                fdn_fs.amp*=ampfs
                while indFDnarr<self.FDnarr.size and not self.FDnarr[indFDnarr].addedifequal(fdn_fs):
                    indFDnarr+=1
                if indFDnarr==self.FDnarr.size:
                    self.FDnarr=np.append(self.FDnarr,fdn_fs)
        self.fsources=[fdn.givefsourcewoamp(z0=z,nofsigranges=np.array([]),longphase=fsource.longphase) for fdn in self.FDnarr]
        self.amps=np.array([fdn.givefsourceamp() for fdn in self.FDnarr])
    
    
    def printfockrepatz(self,z,rounddig=4,showy=False,absamps=False,projns=[],projxs=[],projys=[],demolish=False,etheline=True):
        """print Fock representation at z point"""
        if etheline:
            print('the Fock representation of the state is:')
        if z !=self.zFDnarr:
            self.ExpandSimplifyatz(z=z)
        norm=0
        totnorm=0
        for ifs in range(len(self.fsources)):
            ampmultiplier=self.amps[ifs]
            norm0 ,totnorm0 =self.fsources[ifs].printfockrepatz(z=z,rounddig=rounddig,showy=showy,absamps=absamps,projns=projns,projxs=projxs,projys=projys,ampmultiplier=ampmultiplier,demolish=demolish,etheline=False)
            norm+=norm0*abs(ampmultiplier)**2
            totnorm+=totnorm0
        if etheline:
            print('\n')
        print('norm=',norm,'totnorm=',totnorm)
        return norm, totnorm

    def intensityx(self,xs):
        #redundancy exists here! 
        # every elements is calculated two times, i.e., itself and its complex conjugate.
        intensityatxs=np.zeros_like(xs, dtype=complex)
        fsize=len(self.fsources)
        for ind_bra in range(fsize):
            for ind_ket in range(fsize):
                amp2=self.amps[ind_bra].conjugate()*self.amps[ind_ket]
                intensityatxs+=amp2*FSource.intensityx_sandwiche(xs=xs,bra_fsource=self.fsources[ind_bra],ket_fsource=self.fsources[ind_ket])
        return intensityatxs

    def intensityxtest(self,xs):
        return self.mycheckintensity
        
    
    def setxyranges(self):
        self.fsources[0].setxyranges()

        self.minx=self.fsources[0].minx
        self.maxx=self.fsources[0].maxx
        self.minkx=self.fsources[0].minkx
        self.maxkx=self.fsources[0].maxkx

        self.miny=self.fsources[0].miny
        self.maxy=self.fsources[0].maxy
        self.minky=self.fsources[0].minky
        self.maxky=self.fsources[0].maxky

        for s in self.fsources:
            s.setxyranges()
            self.minx=min(self.minx,s.minx)
            self.maxx=max(self.maxx,s.maxx)
            self.minkx=min(self.minkx,s.minkx)
            self.maxkx=max(self.maxkx,s.maxkx)

            self.miny=min(self.miny,s.miny)
            self.maxy=max(self.maxy,s.maxy)
            self.minky=min(self.minky,s.minky)
            self.maxky=max(self.maxky,s.maxky)

    def setranges(self, zmin, zmax):
        self.zmin=zmin
        self.zmax=zmax
        self.fsources[0].setranges(zmin, zmax)

        self.minx=self.fsources[0].minx
        self.maxx=self.fsources[0].maxx
        self.minkx=self.fsources[0].minkx
        self.maxkx=self.fsources[0].maxkx

        self.miny=self.fsources[0].miny
        self.maxy=self.fsources[0].maxy
        self.minky=self.fsources[0].minky
        self.maxky=self.fsources[0].maxky

        for s in self.fsources:
            s.setranges(zmin, zmax)
            self.minx=min(self.minx,s.minx)
            self.maxx=max(self.maxx,s.maxx)
            self.minkx=min(self.minkx,s.minkx)
            self.maxkx=max(self.maxkx,s.maxkx)

            self.miny=min(self.miny,s.miny)
            self.maxy=max(self.maxy,s.maxy)
            self.minky=min(self.minky,s.minky)
            self.maxky=max(self.maxky,s.maxky)
        

    def goto(self, z):
        for s in range(len(self.fsources)):#(self.noffs):
            self.fsources[s].goto(z)
    def passlens(self, lens):
        for s in range(len(self.fsources)):#range(self.noffs):
            self.fsources[s].passlens(lens)
    def passslmpix(self,slmpix):
         for s in range(len(self.fsources)):#range(self.noffs):
            self.fsources[s].passslmpix(slmpix)
    def passslm(self, slm):
        for s in range(len(self.fsources)):#range(self.noffs):
            self.fsources[s].passslm(slm)

class FSource(qstate): # factorized version
    def __init__(self,bsources,nbsources=[],dx=1,dy=1,bsourcesold=[],activeold=False):
        self.dx=dx #dx = distance between discretized wavefront i=x/dx for |n_{i,j}>
        self.dy=dy #dy = distance between discretized wavefront j=y/dy for |n_{i,j}>
        # Note: DO NOT USE beam relate info from source, such as source.amps, source.x0s, ....
        nofbs=len(bsources) # number of sources
        self.longphase=bsources[0].longphase  #longphase=longitudinal phase
        self.bsources=[copy.deepcopy(bs) for bs in bsources]
        self.nbsources=np.full((nofbs), 1,dtype=int) if len(nbsources)==0 else nbsources
        self.bsourcesold=[copy.deepcopy(bs) for bs in bsourcesold]
        self.activeold=activeold
        qstate.__init__(self)

    
    def setxyranges(self):
        self.bsources[0].setxyranges()

        self.minx=self.bsources[0].minx
        self.maxx=self.bsources[0].maxx
        self.minkx=self.bsources[0].minkx
        self.maxkx=self.bsources[0].maxkx

        self.miny=self.bsources[0].miny
        self.maxy=self.bsources[0].maxy
        self.minky=self.bsources[0].minky
        self.maxky=self.bsources[0].maxky

        for s in self.bsources:
            self.bsources[0].setxyranges()
            self.minx=min(self.minx,s.minx)
            self.maxx=max(self.maxx,s.maxx)
            self.minkx=min(self.minkx,s.minkx)
            self.maxkx=max(self.maxkx,s.maxkx)

            self.miny=min(self.miny,s.miny)
            self.maxy=max(self.maxy,s.maxy)
            self.minky=min(self.minky,s.minky)
            self.maxky=max(self.maxky,s.maxky)

    def setranges(self, zmin, zmax):
        self.zmin=zmin
        self.zmax=zmax
        self.bsources[0].setranges(zmin, zmax)

        self.minx=self.bsources[0].minx
        self.maxx=self.bsources[0].maxx
        self.minkx=self.bsources[0].minkx
        self.maxkx=self.bsources[0].maxkx

        self.miny=self.bsources[0].miny
        self.maxy=self.bsources[0].maxy
        self.minky=self.bsources[0].minky
        self.maxky=self.bsources[0].maxky

        for s in self.bsources:
            s.setranges(zmin, zmax)
            self.minx=min(self.minx,s.minx)
            self.maxx=max(self.maxx,s.maxx)
            self.minkx=min(self.minkx,s.minkx)
            self.maxkx=max(self.maxkx,s.maxkx)

            self.miny=min(self.miny,s.miny)
            self.maxy=max(self.maxy,s.maxy)
            self.minky=min(self.minky,s.minky)
            self.maxky=max(self.maxky,s.maxky)
        

    def goto(self, z):
        nofbs=len(self.bsources)
        for s in range(nofbs):
            self.bsources[s].goto(z)
        if self.activeold:
            for s in range(len(self.bsourcesold)):
                self.bsourcesold[s].goto(z)
    def passlens(self, lens):
        nofbs=len(self.bsources)
        for s in range(nofbs):
            self.bsources[s].passlens(lens)
        if self.activeold:
            for s in range(len(self.bsourcesold)):
                self.bsourcesold[s].passlens(lens)
    def passslmpix(self,slmpix):
        nofbs=len(self.bsources)
        for s in range(nofbs):
            self.bsources[s].passslmpix(slmpix)
        if self.activeold:
            for s in range(len(self.bsourcesold)):
                self.bsourcesold[s].passslmpix(slmpix)
    def passslm(self, slm):
        nofbs=len(self.bsources)
        for s in range(nofbs):
            self.bsources[s].passslm(slm)
        if self.activeold:
            for s in range(len(self.bsourcesold)):
                self.bsourcesold[s].passslm(slm)
    def setfockrepatz(self,z):
        """set Fock representation at z point"""
        bsource=self.bsources[0]
        nbsource=self.nbsources[0]
        x0ss,y0ss,ampss,kx0ss,ky0ss,sigxss,sigyss = bsource.getampsatz(z)
        FDnarr=np.empty([ampss.size],dtype=FDn)
        
        for ia in range(ampss.size):
            newfdn=FDn(dx=self.dx,dy=self.dy)
            newfdn.applyad(k=bsource.k,x0=x0ss[ia],y0=y0ss[ia],kx0=kx0ss[ia],ky0=ky0ss[ia],sigx=sigxss[ia],sigy=sigyss[ia],amp=ampss[ia],nofad=1)
            FDnarr[ia]=newfdn
        for nofad in range(1,nbsource):
            preFDnarr=copy.deepcopy(FDnarr)
            FDnarr=np.array([],dtype=FDn)
            for ia in range(ampss.size):
                adjustFDnarr=copy.deepcopy(preFDnarr)
                for ajfdn in adjustFDnarr:
                    ajfdn.applyad(k=bsource.k,x0=x0ss[ia],y0=y0ss[ia],kx0=kx0ss[ia],ky0=ky0ss[ia],sigx=sigxss[ia],sigy=sigyss[ia],amp=ampss[ia],nofad=1)

                    indFDnarr=0
                    while indFDnarr<FDnarr.size and not FDnarr[indFDnarr].addedifequal(ajfdn):
                        indFDnarr+=1
                    if indFDnarr==FDnarr.size:
                        FDnarr=np.append(FDnarr,ajfdn)
        nofbs=len(self.bsources)
        for bs in range(1,nofbs):
            bsource=self.bsources[bs]
            nbsource=self.nbsources[bs]
            x0ss,y0ss,ampss,kx0ss,ky0ss,sigxss,sigyss = bsource.getampsatz(z)
            for nofad in range(0,nbsource):
                preFDnarr=copy.deepcopy(FDnarr)
                FDnarr=np.array([],dtype=FDn)
                for ia in range(ampss.size):
                    adjustFDnarr=copy.deepcopy(preFDnarr)
                    for ajfdn in adjustFDnarr:
                        ajfdn.applyad(k=bsource.k,x0=x0ss[ia],y0=y0ss[ia],kx0=kx0ss[ia],ky0=ky0ss[ia],sigx=sigxss[ia],sigy=sigyss[ia],amp=ampss[ia],nofad=1)

                        indFDnarr=0
                        while indFDnarr<FDnarr.size and not FDnarr[indFDnarr].addedifequal(ajfdn):
                            indFDnarr+=1
                        if indFDnarr==FDnarr.size:
                            FDnarr=np.append(FDnarr,ajfdn)
        self.zFDnarr=z
        self.FDnarr=FDnarr
    def projectFDnarr(self,z,projns,projxs,projys,demolish=False):
        self.setfockrepatz(z=z)
        norm=0
        totnorm=0
        fdnarr=np.array([],dtype=FDn)
        for fdn in self.FDnarr:
            projected=fdn.project(projns=projns,projxs=projxs,projys=projys,demolish=demolish)
            if projected:
                if demolish:
                    fdn.removezeros()
                if fdn.totn!=0:#fdn.Dns.size()!=0:
                    fdnarr=np.append(fdnarr,fdn)
                norm+= abs(fdn.amp)**2
            totnorm+= abs(fdn.amp)**2
        self.FDnarr=fdnarr
        return norm ,totnorm
    #@jit(parallel=True)#(nopython= True)
    def printfockrepatz(self,z,rounddig=4,showy=False,absamps=False,projns=[],projxs=[],projys=[],ampmultiplier=1,demolish=False,etheline=True):
        """print Fock representation at z point"""
        self.setfockrepatz(z=z)
        if etheline:
            print('the Fock representation of the state is:')
        norm=0
        totnorm=0
        for fdn in self.FDnarr:
            #print('\n +')
            if len(projns)==0:
                fdn.printbase(rounddig=rounddig,showy=showy,absamps=absamps,ampmultiplier=ampmultiplier)
                norm+= abs(fdn.amp)**2
            else:
                projected=fdn.printbaseifprojected(rounddig=rounddig,showy=showy,absamps=absamps,projns=projns,projxs=projxs,projys=projys,ampmultiplier=ampmultiplier,demolish=demolish)
                if projected:
                    norm+= abs(fdn.amp)**2
            totnorm+= abs(fdn.amp)**2
        if etheline:
            print('\n')
        if False:    
            print('norm=',norm,'totnorm=',totnorm)
        return norm ,totnorm

    def intensityxdiff(self,xs):
        out1=self.intensityx(xs)
        out2=self.intensityxold(xs)
        return out1-out2
    def intensityx_sandwiche(xs,bra_fsource,ket_fsource):   #ax ad_{nofbs-1} ..... ad_{0}  |0>   =>   the number of operators at right sided is nofbs+1
        
        #there are redundancy in this function:
        #a suggestion:  ax ad_1^n1 ad_^n2 ...ad_^nm= n1 xi1(x) ad^{n1-1} ad_^n2 ..ad_^nm+ n2 xi2(x) ad^n1 ad_^{n2-1} ..ad_^nm+...
        
        BSpwprods=giveinnprodBSofFS(bra_bsources=bra_fsource.bsources,ket_bsources=ket_fsource.bsources) #BSpwprod=pairwise products of Basic Sources
        if np.sum(bra_fsource.nbsources)==np.sum(ket_fsource.nbsources):
            nofallbs=np.sum(bra_fsource.nbsources)
        else:
            print('take care! the inner prod of two number state with different ns is zero')
            err=1/0  ###
            return 0
        getpindbsext_bra=np.empty(nofallbs,dtype=int) # get pure index of basic sources at extended ragime
        getpindbsext_ket=np.empty(nofallbs,dtype=int) # get pure index of basic sources at extended ragime
        indbsext=0
        nofbs_bra=len(bra_fsource.bsources)
        for pindbs in range(nofbs_bra): #pure index of basic sources
            for i in range(bra_fsource.nbsources[pindbs]):
                getpindbsext_bra[indbsext]=pindbs
                indbsext+=1
        indbsext=0
        nofbs_ket=len(ket_fsource.bsources)
        for pindbs in range(nofbs_ket): #pure index of basic sources
            for i in range(ket_fsource.nbsources[pindbs]):
                getpindbsext_ket[indbsext]=pindbs
                indbsext+=1
                
        totbs=nofallbs+1 # the extra 1 is for ax operator; 
        permsforbra = permutations(range(totbs))
        out_ampsatxs=np.zeros_like(xs,dtype=complex)
        for mypermofbra in permsforbra:
            if mypermofbra[nofallbs]!=nofallbs:
               # we need to ignore the cases where a_x and ad_x are at the original position
               # because ax will be paired with mypermofbra[ketnofbs]
                myperminnprod=1
                for indbsext in range(nofallbs):
                    ket_pind=getpindbsext_ket[indbsext]
                    braindbsext=mypermofbra[indbsext]
                    if braindbsext!=nofallbs:
                        bra_pind=getpindbsext_bra[braindbsext] #bra pure index
                        myperminnprod*=BSpwprods[bra_pind,ket_pind]
                    else:
                        ketpsiind=ket_pind #finally we found adx is paried with which bs; it is paired with self.bsources[psibraind]
                        brapsiindext=mypermofbra[nofallbs]
                        brapsiind=getpindbsext_bra[brapsiindext]
                if abs(myperminnprod)> 0.0001:
                    out_ampsatxs+=myperminnprod*np.multiply(np.conj(bra_fsource.bsources[brapsiind].wpx(xs)),ket_fsource.bsources[ketpsiind].wpx(xs)) 
        return out_ampsatxs
    def intensityxnew(self,xs):   #ax ad_{nofbs-1} ..... ad_{0}  |0>   =>   the number of operators at right sided is nofbs+1
        return FSource.intensityx_sandwiche(xs,bra_fsource=self,ket_fsource=self)
        #there are redundancy in this function:
        #a suggestion:  ax ad_1^n1 ad_^n2 ...ad_^nm= n1 xi1(x) ad^{n1-1} ad_^n2 ..ad_^nm+ n2 xi2(x) ad^n1 ad_^{n2-1} ..ad_^nm+...

    def intensityx(self,xs):   #ax ad_{nofbs-1} ..... ad_{0}  |0>   =>   the number of operators at right sided is nofbs+1   
        #there are redundancy in this function:
        #a suggestion:  ax ad_1^n1 ad_^n2 ...ad_^nm= n1 xi1(x) ad^{n1-1} ad_^n2 ..ad_^nm+ n2 xi2(x) ad^n1 ad_^{n2-1} ..ad_^nm+...
        
        BSpwprods=giveinnprodBSofFS(bra_bsources=self.bsources,ket_bsources=self.bsources) #BSpwprod=pairwise products of Basic Sources
        nofallbs=np.sum(self.nbsources)
        getpindbsext=np.empty(nofallbs,dtype=int) # get pure index of basic sources at etended ragime
        indbsext=0
        nofbs=len(self.bsources)
        for pindbs in range(nofbs): #pure index of basic sources
            for i in range(self.nbsources[pindbs]):
                getpindbsext[indbsext]=pindbs
                indbsext+=1
                

        totbs=nofallbs+1 # the extra 1 is for ax operator; 
        permsforbra = permutations(range(totbs))
        out_ampsatxs=np.zeros_like(xs,dtype=complex)
        for mypermofbra in permsforbra:
            if mypermofbra[nofallbs]!=nofallbs:
               # we need to ignore the cases where a_x and ad_x are at the original position
               # because ax will be paired with mypermofbra[ketnofbs]
                myperminnprod=1
                for indbsext in range(nofallbs):
                    pind=getpindbsext[indbsext]
                    braindbsext=mypermofbra[indbsext]
                    if braindbsext!=nofallbs:
                        bra_pind=getpindbsext[braindbsext] #bra pure index
                        myperminnprod*=BSpwprods[bra_pind,pind]
                    else:
                        ketpsiind=pind #finally we found adx is paried with which bs; it is paired with self.bsources[psibraind]
                        brapsiindext=mypermofbra[nofallbs]
                        brapsiind=getpindbsext[brapsiindext]
                if abs(myperminnprod)> 0.0001:
                    out_ampsatxs+=myperminnprod*np.multiply(np.conj(self.bsources[brapsiind].wpx(xs)),self.bsources[ketpsiind].wpx(xs)) 
        return out_ampsatxs

    def intensityxold(self,xs):   #ax ad_{nofbs-1} ..... ad_{0}  |0>   =>   the number of operators at right sided is nofbs+1
        BSpwprods=giveinnprodBSofFS(bra_bsources=self.bsourcesold,ket_bsources=self.bsourcesold) #BSpwprod=pairwise products of Basic Sources
        nofbs=len(self.bsourcesold)
        totbs=nofbs+1 # the extra 1 is for ax operator; 
        permsforbra = permutations(range(totbs))
        out_ampsatxs=np.zeros_like(xs,dtype=complex)
        for mypermofbra in permsforbra:
            if mypermofbra[nofbs]!=nofbs:
               # we need to ignore the cases where a_x and ad_x are at the original position
               # because ax will be paired with mypermofbra[ketnofbs]
                myperminnprod=1
                for ind in range(nofbs):
                    braind=mypermofbra[ind]
                    if braind!=nofbs:
                        myperminnprod*=BSpwprods[braind,ind]
                    else:
                        ketpsiind=ind #finally we found adx is paried with which bs; it is paired with self.bsources[psibraind]
                if abs(myperminnprod)> 0.0001:
                    out_ampsatxs+=myperminnprod*np.multiply(np.conj(self.bsourcesold[mypermofbra[nofbs]].wpx(xs)),self.bsourcesold[ketpsiind].wpx(xs)) 
        return out_ampsatxs
    
    def intensitykx(self,kxs):   #a_x ad_{nofbs-1} ..... ad_{0}  |0> =>   the number of operators at right sided is nofbs+1
        BSpwprods=giveinnprodBSofFS(bra_bsources=self.bsourcesold,ket_bsources=self.bsourcesold) #BSpwprod=pairwise products of Basic Sources
        ketnofbs=len(self.bsourcesold)
        perm = permutations(range(ketnofbs+1))
        outamps=np.zeros_like(kxs,dtype=complex)
        for myperm in perm:
            if myperm[ketnofbs]!=ketnofbs:
                myperminnprod=1
                for ind in range(ketnofbs):
                    braind=myperm[ind]
                    if braind!=ketnofbs:
                        myperminnprod*=BSpwprods[braind,ind]
                    else:
                        ketpsiind=ind
                if abs(myperminnprod)> 0.0001:
                    outamps+=myperminnprod*np.multiply(np.conj(self.bsourcesold[myperm[ketnofbs]].wpkx(kxs)),self.bsourcesold[ketpsiind].wpkx(kxs)) 
        return outamps