from xpp2python import *
from model import *
from dynamics import *

def main():

	info=read_info('model.ode.pars')
	
	sol=simul(odefun=odde, tspan=[t0,tend], y0=y0, pars=pars,info=info, fign=1, keep_old=False)
	
	pars_temp=pars.copy()
	pars_temp['inc_i1']=0.75
	pars_temp['it1']=300
	
	sol=simul(odefun=odde, tspan=[t0,tend], y0=y0, pars=pars_temp,info=info, fign=1, keep_old=True)
	
	bif_fig(bif_name='bifurcation_data.dat', pars=pars)
	
	bif_fig(bif_name='bifurcation_data.dat', pars=pars, dec_bound=True)
	
	ani=animation(fignum=4)
	
	ani2=animation(fignum=5,dec_bound=True)
	
	outputs(fn=12)
	
	outputs(fn=13)
	return ani, ani2

if __name__=='__main__':
	ani,ani2=main()
	plt.show()