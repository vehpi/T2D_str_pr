'''This module contains functions that take XPP info and dat files as input 
and generate python scripts or data frame. Generated python script is ready 
to be run/integrated on python.
'''

def read_info(fname):
	'''This function reads information from the .ode.pars info file generated by XPP'''
	
	file=open(fname)
	lines_temp=file.readlines()
	file.close()
	
	lines_temp=[i.replace('^','**') for i in lines_temp]
	
	lines=[i.strip().lower() for i in lines_temp]
	lines.append('')
	var=[]
	eq=[]
	num=[]
	pars=[]
	ics=[]
	num_pars=[]
	functions=[]
	
	for i in range(len(lines)):
		if lines[i]=='equations...':
			j=i+1
			while lines[j]!='':
				if '/dt' in lines[j]:
					var.append(lines[j].split('/dt',1)[0][1:])
					j=j+1
				else:
					var.append(lines[j].split('=',1)[0])
					j=j+1
		if lines[i]=='equations...':
			j=i+1
			while lines[j]!='':
				if '/dt' in lines[j]:
					eq.append(lines[j].replace('/dt',''))
					j=j+1
				else:
					break
		if lines[i]=='where ...':
			j=i+1
			while lines[j]!='':
				num.append(lines[j])
				j=j+1
		if lines[i]=='user-defined functions:':
			j=i+1
			while lines[j]!='':
				functions.append(lines[j])
				j=j+1
		if lines[i]=='ics ...':
			j=i+1
			while lines[j]!='':
				ics.append(lines[j].split('=',1)[1])
				j=j+1
		if lines[i]=='parameters ...':
			j=i+1
			while lines[j]!='':
				pars.append(lines[j].split())
				j=j+1
	
		if lines[i]=='numerical parameters ...':
			j=i+1
			while lines[j]!='':
				num_pars.append(lines[j].split())
				j=j+1
	
	num_pars=[i for j in num_pars for i in j]
	tspan=[eval(i.split('=',2)[1]) for i in num_pars if ('t0' in i) or ('tend' in i)]
	
	pars=[i for j in pars for i in j]
	dyn_vars=var[0:len(ics)]
	aux_vars=var[len(ics):]
	dy=['d'+i for i in dyn_vars]
	info={'var':var,'eq':eq,'num':num,'functions':functions,'pars':pars,'ics':ics,'num_pars':num_pars,'dyn_vars':dyn_vars,'aux_vars':aux_vars,'dy':dy,'tspan':tspan}
	return info

def create_script(fname,outname=False):
	'''This function creates and writes a python script from the information 
	read from the XPP info file, which is passed to read_info function'''
	infod=read_info(fname)
	var=infod['var']
	eq=infod['eq']
	num=infod['num']
	pars=infod['pars']
	ics=infod['ics']
	num_pars=infod['num_pars']
	dyn_vars=infod['dyn_vars']
	aux_vars=infod['aux_vars']
	dy=infod['dy']
	tspan=infod['tspan']
	functions=infod['functions']
	
	fnames=fname.split('.',2)
	if outname:
		py_f=open(outname+'.py','w')    ######   t0tend
	else:
		py_f=open(fnames[0]+'.py','w')

	py_f.write('#This file has been created from {fname} \n'.format(fname=fname))
	
	py_f.write('import numpy as np \n')
	py_f.write('from numpy import sin, cos, exp, sqrt \n')
	py_f.write('import matplotlib.pyplot as plt \n')
	py_f.write('from scipy.integrate import solve_ivp as sivp'+2*'\n')
	py_f.write('inf=np.inf'+2*'\n')
	
	py_f.write('\n'+'#Functions \n')
	for ftx in functions:
		ftx=ftx.split('=')
		ftx[0]=ftx[0].split('(')
		ftx[0][1]=ftx[0][1].replace(')','')
		py_f.write('\t' + ftx[0][0]+ '=lambda ' + ftx[0][1] + ':' + ftx[1] + '\n')
 
	py_f.write('pars={} \n')
	for pv in pars:
		pn=pv.split('=',2)[0]
		val=pv.split('=',2)[1]
		py_f.write('pars[\''+pn+'\']='+val + '\n')
	
	py_f.write('\n')
	py_f.write('pars_l=list(pars.values()) \n')
	py_f.write('pars_npa=np.array(pars_l) \n')
	py_f.write('pars_n=list(pars.keys()) \n')
	
	py_f.write('\n')
	py_f.write('\n'+'def odde(t,y,pars_npa): \n')
	
	py_f.write('\n'+'\t'+'heav=lambda x: np.heaviside(x,1) \n')
	
	py_f.write('\n #Initial Values \n')
	for i,j in enumerate(dyn_vars):
		py_f.write('\t'+j+'='+'y[{i}]'.format(i=i)+'\n')
	
	py_f.write('\n #Parameter Values \n')
	for i,pv in enumerate(pars):
		pn=pv.split('=',2)[0]
		py_f.write('\t'+ '{pn1}=pars_npa[{i}]'.format(pn1=pn,i=i) +'\n')
		
	py_f.write('\n'+'#Numerics \n')	
	for i in num:
		py_f.write('\t'+i+'\n')
		
	py_f.write('\n'+'#Diferetial Equations \n')	
	for i in eq:
		py_f.write('\t'+i+'\n')
	
	py_f.write('\n'+'\t'+'dy='+'[')
	for i in dy:
		if i!=dy[-1]:
			py_f.write(i+',')
		else:
			py_f.write(i)
	py_f.write(']'+'\n')	
	
	py_f.write('\n'+'\t'+'return dy')
	
	py_f.write('\n'+'y0='+'[')
	for i,j in enumerate(ics):
		if i<len(ics)-1:
			py_f.write(j+',')
		else:
			py_f.write(j)
	py_f.write(']'+'\n')
	
	py_f.write(f't0={tspan[0]}'+'\n')
	py_f.write(f'tend={tspan[1]}'+'\n')
	py_f.write('if __name__=="__main__":'+'\n')
	py_f.write('\t'+'sol=sivp(odde,[t0,tend],y0,method=\'LSODA\',args=[pars_npa])')
	
	py_f.close()
	return infod

def simul(odefun, tspan, y0, pars, info=None, fign=123, keep_old=False, draw_fig=True):
	from scipy.integrate import solve_ivp as sivp
	import matplotlib.pyplot as plt
	import numpy as np
	
	pars_npa=np.array(list(pars.values()))
	
	sol=sivp(odefun,t_span=tspan,y0=y0,method='LSODA',args=[pars_npa])
	if draw_fig:
		if info:
			vn=info['dyn_vars']
		else:
			vn=[str(i) for i in range(len(sol.y))]
		
		if keep_old:
			fig=plt.figure(fign)
			axs=fig.get_axes()
		else:
			fig,axs=plt.subplots((len(vn)+1)//2,2)
			axs=axs.ravel()
		for i,ax in enumerate(axs):
			ax.plot(sol.t/30,sol.y[i])
			ax.set_xlabel('Time (month)')
			ax.set_ylabel(vn[i])
			ax.set_ylim([0.9*sol.y[i].min(),1.1*sol.y[i].max()])
		
		fig.set_size_inches([7.1, 7.59])
		fig.subplots_adjust(top=0.97,
						 bottom=0.075,
						 left=0.1,
						 right=0.955,
						 hspace=0.55,
						 wspace=0.27)
		fig.show()
	return sol
