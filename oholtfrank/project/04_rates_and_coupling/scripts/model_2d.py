#!/usr/bin/env python3
"""Does the Ar-O contraction accelerate VPD?  V(R)=V0*(3.5/R)^3, Gamma~R^-6."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HA2EV=27.2114; ANG2BOHR=1.8897259886; AU2FS=0.0241888
MU=1728.0; OMEGA_CM=3857.09; DE_EV=5.11; RE=0.9572
E_PHOTON=13.48; V0=9.352e-3; B_REP=2.6
omega_au=OMEGA_CM/219474.63
a_ang=(omega_au*np.sqrt(MU/(2*DE_EV/HA2EV)))*ANG2BOHR
def VD_eV(r): return DE_EV*(1-np.exp(-a_ang*(r-RE)))**2
VI=-(E_PHOTON-DE_EV); A=-VI
def VA_eV(r): return A*np.exp(-B_REP*(r-RE))+VI
def R_of_t(t): return np.where(t<100.0, 3.48-1.00*(t/100.0), 2.48)

N=2048; rA=np.linspace(0.4,12.0,N); r=rA*ANG2BOHR
dr=r[1]-r[0]; k=2*np.pi*np.fft.fftfreq(N,d=dr)
VD=VD_eV(rA)/HA2EV; VA=VA_eV(rA)/HA2EV
ab=np.ones(N); e=rA>10.0; ab[e]=np.exp(-((rA[e]-10.0)/1.0)**2)

def run(fixed=None, tmax=300.0):
    alpha=MU*omega_au
    psiD=((alpha/np.pi)**0.25*np.exp(-0.5*alpha*(r-RE*ANG2BOHR)**2)).astype(complex)
    psiD/=np.sqrt(np.sum(np.abs(psiD)**2)*dr); psiA=np.zeros(N,dtype=complex)
    dt=1.0; nst=int(tmax/AU2FS/dt); Tk=np.exp(-1j*(k**2/(2*MU))*dt/2)
    ts=[];pa=[];acc=0.0
    for n in range(nst):
        t=n*dt*AU2FS
        R = fixed if fixed else float(R_of_t(np.array([t]))[0])
        V12=(V0*(3.5/R)**3)/HA2EV
        dV=0.5*(VD-VA); av=0.5*(VD+VA); Om=np.sqrt(dV**2+V12**2)
        c_=np.cos(Om*dt);s_=np.sin(Om*dt);ph=np.exp(-1j*av*dt)
        U11=ph*(c_-1j*s_*dV/Om);U22=ph*(c_+1j*s_*dV/Om);U12=ph*(-1j*s_*V12/Om)
        psiD=np.fft.ifft(Tk*np.fft.fft(psiD));psiA=np.fft.ifft(Tk*np.fft.fft(psiA))
        nD=U11*psiD+U12*psiA; nA=U12*psiD+U22*psiA; psiD,psiA=nD,nA
        psiD=np.fft.ifft(Tk*np.fft.fft(psiD));psiA=np.fft.ifft(Tk*np.fft.fft(psiA))
        b=(np.sum(np.abs(psiD)**2)+np.sum(np.abs(psiA)**2))*dr
        psiD*=ab;psiA*=ab
        acc+=b-(np.sum(np.abs(psiD)**2)+np.sum(np.abs(psiA)**2))*dr
        if n%100==0: ts.append(t); pa.append(np.sum(np.abs(psiA)**2)*dr+acc)
    return np.array(ts),np.array(pa)

print("fixed R=3.5 A ..."); t1,p1=run(fixed=3.5)
print("contracting R(t) ..."); t2,p2=run()
print("\n transferred at 300 fs:")
print("   fixed R=3.5 A : %.4e"%p1[-1])
print("   contracting   : %.4e"%p2[-1])
print("   enhancement   : %.2f x"%(p2[-1]/p1[-1]))
print("\n naive R^-6 at 2.48 A: %.1f x"%((3.48/2.48)**6))
fig,ax=plt.subplots(1,2,figsize=(11,4))
tt=np.linspace(0,300,300)
ax[0].plot(tt,R_of_t(tt),'g-',lw=2); ax[0].axhline(3.5,ls=':',c='gray')
ax[0].set_xlabel('t (fs)'); ax[0].set_ylabel(r'$R$(Ar-O) [$\AA$]')
ax[0].set_title('AIMS trajectory 1')
ax[1].semilogy(t1,p1+1e-300,'k--',lw=2,label='fixed 3.5 $\\AA$')
ax[1].semilogy(t2,p2+1e-300,'r-',lw=2,label='contracting')
ax[1].set_xlabel('t (fs)'); ax[1].set_ylabel(r'$P_{acceptor}$'); ax[1].legend(fontsize=8)
ax[1].set_title('dynamical coupling enhancement')
plt.tight_layout(); plt.savefig('fig_2d_enhancement.png',dpi=150)
print("wrote fig_2d_enhancement.png")
