#This file has been created from model.ode.pars 
import numpy as np 
from numpy import sin, cos, exp, sqrt 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp as sivp

inf=np.inf

pars={} 
pars['eg0']=24.48
pars['k']=700
pars['bv']=5
pars['mmax']=1
pars['alpha_m']=140
pars['km']=2
pars['alpha_isr']=1.2
pars['kisr']=2
pars['pmax']=4.55
pars['kp']=4
pars['alpha_p']=35
pars['p_b']=0
pars['amax']=5
pars['alpha_a']=0.37
pars['ka']=6
pars['a_b']=0.9
pars['tau_b']=1800
pars['height']=1.8
pars['age_b']=30
pars['cage']=0
pars['target_si']=1.4
pars['tau_si']=1
pars['bmi_h']=25
pars['mffa']=0.8
pars['ksi_infl']=1.8
pars['ksi_ffa']=400
pars['nsi_ffa']=6
pars['nsi_infl']=1
pars['inc_i1']=0
pars['inc_i2']=0
pars['inc_i3']=0
pars['it1']=0
pars['it2']=inf
pars['it3']=inf
pars['kpc']=11550
pars['tau_w']=1
pars['k_infl']=40
pars['n_infl']=6
pars['infl_b']=0
pars['tau_infl']=1
pars['hgp_bas']=2000
pars['hepa_max']=3000
pars['hepa_sic']=1
pars['a_hgp']=4
pars['gcg']=0.1
pars['k_gcg']=0
pars['s']=0.0002
pars['png']=350
pars['sigma_b']=536
pars['sgmu']=1.5
pars['sgmd']=1
pars['sfm']=1.2
pars['sim']=0.25
pars['sgku']=81
pars['sgkd']=137
pars['sfk']=357
pars['sik']=0.6
pars['nsgku']=6
pars['nsgkd']=6
pars['nsfk']=6
pars['nsik']=4
pars['tau_sigma']=1
pars['sci']=1
pars['gclamp']=0
pars['iclamp']=0
pars['maxinfl']=1
pars['l0']=170
pars['l2']=8.1
pars['cf']=2
pars['ksif']=11
pars['aa']=2
pars['sif']=1
pars['ffa_ramax']=1.106

pars_l=list(pars.values()) 
pars_npa=np.array(pars_l) 
pars_n=list(pars.keys()) 


def odde(t,y,pars_npa): 

	heav=lambda x: np.heaviside(x,1) 

 #Initial Values 
	g=y[0]
	i=y[1]
	ffa=y[2]
	si=y[3]
	b=y[4]
	sigma=y[5]
	infl=y[6]
	w=y[7]

 #Parameter Values 
	eg0=pars_npa[0]
	k=pars_npa[1]
	bv=pars_npa[2]
	mmax=pars_npa[3]
	alpha_m=pars_npa[4]
	km=pars_npa[5]
	alpha_isr=pars_npa[6]
	kisr=pars_npa[7]
	pmax=pars_npa[8]
	kp=pars_npa[9]
	alpha_p=pars_npa[10]
	p_b=pars_npa[11]
	amax=pars_npa[12]
	alpha_a=pars_npa[13]
	ka=pars_npa[14]
	a_b=pars_npa[15]
	tau_b=pars_npa[16]
	height=pars_npa[17]
	age_b=pars_npa[18]
	cage=pars_npa[19]
	target_si=pars_npa[20]
	tau_si=pars_npa[21]
	bmi_h=pars_npa[22]
	mffa=pars_npa[23]
	ksi_infl=pars_npa[24]
	ksi_ffa=pars_npa[25]
	nsi_ffa=pars_npa[26]
	nsi_infl=pars_npa[27]
	inc_i1=pars_npa[28]
	inc_i2=pars_npa[29]
	inc_i3=pars_npa[30]
	it1=pars_npa[31]
	it2=pars_npa[32]
	it3=pars_npa[33]
	kpc=pars_npa[34]
	tau_w=pars_npa[35]
	k_infl=pars_npa[36]
	n_infl=pars_npa[37]
	infl_b=pars_npa[38]
	tau_infl=pars_npa[39]
	hgp_bas=pars_npa[40]
	hepa_max=pars_npa[41]
	hepa_sic=pars_npa[42]
	a_hgp=pars_npa[43]
	gcg=pars_npa[44]
	k_gcg=pars_npa[45]
	s=pars_npa[46]
	png=pars_npa[47]
	sigma_b=pars_npa[48]
	sgmu=pars_npa[49]
	sgmd=pars_npa[50]
	sfm=pars_npa[51]
	sim=pars_npa[52]
	sgku=pars_npa[53]
	sgkd=pars_npa[54]
	sfk=pars_npa[55]
	sik=pars_npa[56]
	nsgku=pars_npa[57]
	nsgkd=pars_npa[58]
	nsfk=pars_npa[59]
	nsik=pars_npa[60]
	tau_sigma=pars_npa[61]
	sci=pars_npa[62]
	gclamp=pars_npa[63]
	iclamp=pars_npa[64]
	maxinfl=pars_npa[65]
	l0=pars_npa[66]
	l2=pars_npa[67]
	cf=pars_npa[68]
	ksif=pars_npa[69]
	aa=pars_npa[70]
	sif=pars_npa[71]
	ffa_ramax=pars_npa[72]

#Numerics 
	bmi = w/height**2
	age = age_b+cage*t/365
	tsi = target_si*(ksi_infl**nsi_infl/(ksi_infl**nsi_infl+infl**nsi_infl))*(1-mffa*ffa**nsi_ffa/(ffa**nsi_ffa+ksi_ffa**nsi_ffa))
	intake_i = 2500
	inc_int =  0 + heav(t-it1)*heav(it2-t)*heav(it3-t)*inc_i1 + heav(t-it2)*heav(it3-t)*inc_i2 + heav(t-it3)*inc_i3
	w_base = 81
	expd = 2500/(kpc*w_base)
	k_w = 1/kpc
	m = mmax*g**km/(alpha_m**km + g**km)
	isr =  sigma*(m)**kisr/(alpha_isr**kisr + (m)**kisr)
	p_isr = p_b + pmax*isr**kp/(alpha_p**kp + isr**kp)
	a_m = a_b + amax*m**ka/(alpha_a**ka + m**ka)
	p =  p_isr
	a =  a_m
	fmass_p = 1.2*bmi + 0.23*age - 16.2
	fmass = w*fmass_p/100
	inc = 1+inc_int
	hepa_si = hepa_sic*si
	hgp = (hgp_bas + hepa_max*(a_hgp+k_gcg*gcg)/((a_hgp+k_gcg*gcg)+hepa_si*i))
	intake = intake_i*inc
	s_glucu = sgmu*g**nsgku/(g**nsgku+sgku**nsgku)
	s_glucd = sgmd*g**nsgkd/(g**nsgkd+sgkd**nsgkd)
	s_ffa = sfm*ffa**nsfk/(ffa**nsfk+sfk**nsfk)
	s_infl = sim*infl**nsik/(infl**nsik+sik**nsik)
	s_inf = sigma_b*(s_glucu - s_glucd*s_ffa - s_infl)
	siff = si*sif
	cl0 = l0*24*60/bv
	cl2 = l2*24*60/bv
	al0 = 2.45*24*60

#Diferetial Equations 
	dg=gclamp+hgp-(eg0+sci*si*i)*g
	di=iclamp+b*isr/bv-k*i
	dffa=(cl0+cl2*fmass)*(ffa_ramax*(ksif**aa)/(ksif**aa+(siff*i)**aa))-cf*w*ffa
	dsi=(tsi-si)/tau_si
	db=(png+p*b-a*b-s*b**2)/tau_b
	dsigma=(s_inf-sigma)/tau_sigma
	dinfl=infl_b+(maxinfl*bmi**n_infl/(bmi**n_infl+k_infl**n_infl)-infl)/tau_infl
	dw=(k_w*intake-expd*w)/tau_w

	dy=[dg,di,dffa,dsi,db,dsigma,dinfl,dw]

	return dy
y0=[94.09178511014869,9.628547236106586,403.9775020327191,0.7984301525822913,1009.263489805445,530.0333453052366,0.05625177755617077,81]
t0=0
tend=3600
if __name__=="__main__":
	sol=sivp(odde,[t0,tend],y0,method='LSODA',args=[pars_npa])