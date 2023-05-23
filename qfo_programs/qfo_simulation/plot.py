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
import matplotlib.pyplot as plt

phasecmap = plt.get_cmap('hsv')
ampcmap=plt.cm.Greens_r

def allphased3dplotsvs(fig,xs,ys,amps,kind='',xunit='',scalefac='',title='',squarefig=True,cuttoview=False):
    fig.suptitle(title, fontsize=16)
    ygps, xgps = np.meshgrid(ys,xs)            
    ax = fig.add_subplot(1, 1, 1) 
    if squarefig:
        Dx=max(xs)-min(xs)
        Dy=max(ys)-min(ys)
        ax.set_aspect(Dx/Dy)
    
    abamps=abs(amps)
    c=ax.pcolormesh(xgps, ygps, abamps, shading='auto', cmap=ampcmap)
    if cuttoview:     
        minamp=np.min(abamps)
        maxamp=np.max(abamps)
        meanamp=np.mean(abamps)
        minofp=np.max([minamp,meanamp-5])
        maxofp=np.min([maxamp,meanamp+5])
        c.set_clim(minofp,maxofp)
    cbar=fig.colorbar(c,ax=ax)  

def allphased3dplots(fig,xs,ys,amps,kind='',xunit='',scalefac='',title='',squarefig=True,cuttoview=False):    
    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    
    ygps, xgps = np.meshgrid(ys,xs)  
    phased3dplot(ax,xs,ys,amps,kind=kind,xunit=xunit)
    ax = fig.add_subplot(1, 3, 2, projection='3d') 
    phased3dplotv2(ax,xgps,ygps,amps,kind=kind,xunit=xunit)          
    ax = fig.add_subplot(1, 3, 3) 
    if squarefig:
        Dx=max(xs)-min(xs)
        Dy=max(ys)-min(ys)
        ax.set_aspect(Dx/Dy)
    
    abamps=abs(amps)
    c=ax.pcolormesh(xgps, ygps, abamps, shading='auto', cmap=ampcmap)
    if cuttoview:     
        minamp=np.min(abamps)
        maxamp=np.max(abamps)
        meanamp=np.mean(abamps)
        minofp=np.max([minamp,meanamp-5])
        maxofp=np.min([maxamp,meanamp+5])
        c.set_clim(minofp,maxofp)
    cbar=fig.colorbar(c,ax=ax)  
def phasedonlyplot(xs,ys,amps,kind='',xunit='',scalefac='',title=''):
    fig, ax = plt.subplots(constrained_layout=True)

    
    ygps, xgps = np.meshgrid(ys,xs) 

    clim=(-np.pi,np.pi)

    c=ax.pcolormesh(xgps, ygps, np.angle(amps), shading='auto', cmap=phasecmap,clim=clim)
    cbar=fig.colorbar(c,ax=ax,ticks=[-.98*np.pi,0,.98*np.pi],shrink=0.5)
    cbar.ax.set_yticklabels([r'$-\pi$', '0', r' $\pi$'])  # vertically oriented colorbar

    if title != '':
        ax.title.set_text(title)
    pass
    ax.set_xlabel(f'$x ({xunit}) {scalefac}$') 
    ax.set_ylabel(f'$y ({xunit}) {scalefac}$') 
def phased3dplot(ax,xs,ys,amps,kind='',xunit='',scalefac='',title=''):
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ygps, xgps = np.meshgrid(ys,xs)
    scamap = plt.cm.ScalarMappable(cmap=phasecmap)
    phys=np.angle(amps)
    fcolors = scamap.to_rgba(phys)
    minphfac=np.min(phys)/np.pi #-1
    maxphfac=np.max(phys)/np.pi#+1
    clim=(np.pi*minphfac,np.pi*maxphfac)#(-np.pi,np.pi)
    surf = ax.plot_surface(xgps, ygps, abs(amps), facecolors=fcolors, rstride=1, cstride=1,
    linewidth=0, antialiased=False,cmap=phasecmap,clim=clim)
    fig = ax.get_figure() 
    cbar=fig.colorbar(surf,ax=ax,ticks=[minphfac*np.pi,0,maxphfac*np.pi],shrink=0.5)

    cbar.ax.set_yticklabels([f'{minphfac:.1e} $\pi$', '0', f'{maxphfac:.1e} $\pi$'])  # vertically oriented colorbar

    if kind == 'r' :
        ax.set_xlabel(f'$x ({xunit}) {scalefac}$') 
        ax.set_ylabel(f'$y ({xunit}) {scalefac}$') 
        ax.set_zlabel(r'$\xi(x,y) $') 
    elif kind == 'k':
        ax.set_xlabel(f'$k_x ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_ylabel(f'$k_y ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_zlabel(r'$\tilde\xi(k_x,k_y) $')
    elif kind == 'zx' :
        ax.set_xlabel(f'$z ({xunit}) {scalefac}$') 
        ax.set_ylabel(f'$x ({xunit}) {scalefac}$') 
        ax.set_zlabel(r'$\xi(x) $') 
    elif kind == 'zkx':
        ax.set_xlabel(f'$z ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_ylabel(f'$k_x ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_zlabel(r'$\tilde\xi(k_x) $')
    elif kind == 'zy' :
        ax.set_xlabel(f'$z ({xunit}) {scalefac}$') 
        ax.set_ylabel(f'$y ({xunit}) {scalefac}$') 
        ax.set_zlabel(r'$\xi(y) $') 
    elif kind == 'zky':
        ax.set_xlabel(f'$z ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_ylabel(f'$k_y ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_zlabel(r'$\tilde\xi(k_y) $')
    if title != '':
        ax.title.set_text(title)
def phased3dplotv2(ax,xgps,ygps,amps,kind='',xunit='',scalefac='',title=''):
        ax.plot_wireframe(xgps, ygps, abs(amps), rstride=10, cstride=10) 

def plotv2(ax,xs,ys,kind='',xunit='',scalefac='',title='',step=False):
    ampys=ys

    outr= 1.1
    maxy= outr*np.max(ampys)
    minx= xs[0]
    maxx= xs[-1]
    extent=(minx, maxx, 0, maxy) 
    if step:
        ax.step(xs, ampys, 'g^',linestyle='-', where='mid')
    else:
        ax.plot(xs, ampys)

    if kind == 'x' or kind=='y':
        ax.set_xlabel(f'${kind} ({xunit}) {scalefac}$') 
        ax.set_ylabel(fr'$\xi( {kind}) $') 
    elif kind == 'k_x' or kind=='k_y':
        ax.set_xlabel(f'${kind} ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_ylabel(fr'$\tilde\xi({kind}) $')
    if kind == 'phi':
        kindx='x'
        ax.set_xlabel(f'${kindx} ({xunit}) {scalefac}$') 
        ax.set_ylabel(fr'$\phi( {kindx}) $') 
    if title != '':
        ax.title.set_text(title)

def phasedplotv2(ax,xs,ys,kind='',xunit='',scalefac='',title='',minphf=100,maxphf=100):
    ampys=abs(ys)

    phys=np.angle(ys)#%(2*np.pi)
    
    outr= 1.1
    maxy= outr*np.max(ampys)
    minx= xs[0]
    maxx= xs[-1]
    extent=(minx, maxx, 0, maxy) 
    
    ax.plot(xs, ampys)

    sizey=round(outr*ys.size)
    ygps = np.linspace(0, maxy, sizey)
    xcol,Y = np.meshgrid(phys,ygps)

    y = np.linspace(0, maxy, sizey)
    X, Y = np.meshgrid(xs,y)
    c=ax.pcolor(X, Y, xcol, cmap=phasecmap)
    minphfac=-1#np.min(phys)/np.pi if minphf==100 else minphf
    maxphfac=1#np.max(phys)/np.pi if maxphf==100 else maxphf
    c.set_clim(minphfac*np.pi,maxphfac*np.pi)
    fig = ax.get_figure()
    
    cbar=fig.colorbar(c,ax=ax,ticks=[minphfac*np.pi,0,maxphfac*np.pi])
    cbar.ax.set_yticklabels([f'{minphfac:.1e} $\pi$', '0', f'{maxphfac:.1e} $\pi$'])  # vertically oriented colorbar

    ax.fill_between(xs, ampys, maxy, color='w')

    if kind == 'x' or kind=='y':
        ax.set_xlabel(f'${kind} ({xunit}) {scalefac}$') 
        ax.set_ylabel(fr'$\xi( {kind}) $') 
    elif kind == 'k_x' or kind=='k_y':
        ax.set_xlabel(f'${kind} ({xunit}^{{-1}}) {scalefac}$') 
        ax.set_ylabel(fr'$\tilde\xi({kind}) $')
    if title != '':
        ax.title.set_text(title)    