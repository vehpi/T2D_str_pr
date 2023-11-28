import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib import colors, cm
from matplotlib.animation import FuncAnimation, PillowWriter


from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp as sivp
from scipy.interpolate import interp1d

from xpp2python import *
from model import *

def bif_fig(bif_name,pars,fig=None,ax=None,dec_bound=False,ms=100):
	'''
	This function creates a bifurcation diagram from the data exported from XPPAUT AUTO interface.
	bif_name: the name of the bifurcation data file exported from XPPAUT.
	pars: the dictionary that holds model parameters and their values
	fig: optional figure object that can be passed to the function
	ax: optional axis object can be passed to the function
	dec_bound: The option to create a color map  on the phase space that indicates the necessary calorie restriction necessary for remission
	ms: Number of model simulations generated to determine the decision boundries. higher ms results in more accurate decision boundries
	but computationally costly. ms=30 usually works well. Only relevant when dec_bound set True.
	'''
	
	bifd=pd.read_csv(bif_name,delimiter=' ',header=None).values
	
	if not fig:
		fig,ax=plt.subplots(1,1)
	
	
	
	'''
	bifd holds the bifurcation diagram data exported from XPPAUT. 
	bifd[:,0]: the stability of the steady state. 1 for stable, unstable othervise.
	bifd[:,3]: the value of the bifurcation parameter
	bifd[:,[6:6+#num_variables]]: the steady state values of the model variables
	'''
	
	loc1=np.where(bifd[:,3]==50)[0][0]
	bifd=np.append(bifd[1:loc1],np.flip(bifd[loc1:],axis=0),axis=0)
	
	# Sorting the data
	inds=[0]
	inds1=np.linspace(24, 50,200)
	jprev=bifd[0,3]
	b=bifd[:,10]/max(bifd[:,10])
	bprev=bifd[0,10]
	for i,j in enumerate(bifd[:,3]):
		if np.sqrt((j-jprev)**2+(bprev-b[i])**2)>0.02:
			jprev=j
			bprev=b[i]
			inds.append(i)
			
	bifd=bifd[inds]
	
	# bmi is teh bifurcation parameter
	bmi=bifd[:,3]
	ars=np.where(np.logical_and(bmi>=24, bmi<=50))
	
	#using bifurcation parameter values that corresond to BMI between 24 and 50. 
	bifd=bifd[ars[0],:]
	
	g=bifd[:,6]
	i=bifd[:,7]
	ffa=bifd[:,8]
	si=bifd[:9]
	b=bifd[:,10]
	sigma=bifd[:,11]
	infl=bifd[:,12]
	
	x1=bmi
	y1=b
	
	stability=bifd[:,0]	
	#setting color to crimson for stable and to black for unstable branches 
	cmap = ListedColormap(['crimson', 'k'])
	norm = BoundaryNorm([0,1.5,2.5], cmap.N)
	
	# Creating line segments for the diagram
	points = np.array([x1, y1]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	
	#creating LinecCollection object from segments
	lc = LineCollection(segments, cmap=cmap,linestyle='-',linewidth=2)
	lc.set_array(stability)
	lc.set_linewidth(2)
	lc.set_linestyle('-')
	#adding line collection to the axis
	lll=ax.add_collection(lc)
	
	#Setting axis limits and lables
	ax.set_xlim(x1.min()*0.9, x1.max()*1.1)
	ax.set_ylim(y1.min()*0.9, y1.max()*1.1)
	ax.set_ylabel(r'$\beta$ (mg)',size='medium')
	ax.set_xlabel(r'BMI ($kg/m^2$)',size='medium')
	ax.set_xlim([25,50])
	ax.set_ylim([0,2000])
	plt.show()
	
	'''
	Following part is only relevant for creating a color map on the phase space 
	that shows the necessary calorie restriction necessary for remission for the BMI and beta-cell mass (b) values. 
	The unstable branch of the bifurcation diagram serves as the threshold that separates the healthy and diabetic states,
	by separating the basin of attraction for low and high b steady states.
	'''
	if dec_bound:
		
		#We first define branch of unstable steady state bmi (bmius) and b (bbus).
		bmius=bmi[stability!=1]
		bbus=b[stability!=1]
		'''
		We define initial conditions for simulating trajectories. The trajectories should start at maximum allowed bmi (maxbmi)
		and a range of b values, where ms is the number of trajectories to generate. MAx bmi corresponds 
		to a 100% increase in daily enrgy intake. So for each simulation initial inc_i1 is set to 1.
		inc_i1=0 is the baseline that corresponds BMI=25. Sewtting inc_i1 a values below 1 corresponds to a claorie restiriction.
		The percentage of the calorie restriction is 100*(1-inc_i1). Hence lower inc_i1 corresponds to a higher restiriction.
		'''
		minbmi=25
		maxbmi=50
		#converting maximum bmi to weight, which will be used as initial condition for corresponding weight in the model 
		wwa=np.ones(ms)*maxbmi*pars['height']**2
		#determining initial values for a range of b values between 0 and 2000 mg
		bba=np.linspace(0, 2000,ms)
		
		# For each initial condition, simulations are run for diffrerent daily energy intkae restricitions.
		# For a calorie restiriction to be successfull for remission, it must be able to move the 
		# trajectory left side of the branch of unstable steadyy state.
		inc_i1s=[0,0.1,0.2,0.3,0.415,0.5,0.5751]
		
		# interpolating bbus as a function of bmius and vice versa. When interpolating for bbus for values of bmius, 
		# we extrapolate values outside the boundries with the minbmi at the lower bound and max(bmius) at the upper bound.
		# The max(bmius) corresponds to the bmi values at the turning point of the curve, whihc is a saddle node bifurcation.
		f=interp1d(bbus,bmius,bounds_error=False,fill_value=(minbmi,max(bmius)))
		f2=interp1d(bmius,bbus)
		
		# We create empty cols and sols lists. Cols hold 1 for simulations that succedd remission, and 0 those fails.
		# Sols holds the solution structures for later use.
		cols=np.zeros([len(wwa),len(inc_i1s)])
		sols=np.zeros(cols.shape,dtype=object)
		
		# Defining a terminal event function that determines if the solution trajectory passed through the threshold, 
		# namely the branch of unstable steady states.
		def eventf(self,t,y): return t[7]/1.8**2-f(t[4])
		eventf.terminal=True
		
		for j,inc_i1 in enumerate(inc_i1s):
			for i in range(len(wwa)):
				
				jw=wwa[i]
				jb=bba[i]
				''' 
				Determining the initial condition for the solution curves. For each run, 
				we use the jb and jw values as the initial condition for b and w, respectively. 
				We than set the differential equations for these variables to 0 by setting the time constants 
				for these variables (tau_b and tau_w) arbitrarily large values (np.inf). This means these variables would not chnage
				during simulations but the rest of the variables converges to their respective steady state values.
				This is equivalent to fixing b and w to jb and jw respectively and solving the system.
				This step cpould be performed as a root finding problem by using a non-linear solver as well, 
				but we found it to be less convinient, computationally more demanding.
				'''
				y0=np.array([94.1,9.6,403.9,0.8,jb,530,0.056,jw])
				
				tau_w_or=pars['tau_w']
				tau_b_or=pars['tau_b']
				
				pars['tau_b']=np.inf
				pars['tau_w']=np.inf
				pars['inc_i1']=inc_i1
				
				# it1 determines the time of setting daily energy intake to inc_i1, 
				# which is irrelevant for these simulations because w forse not to change 
				# it2 determines the time of making a change in daily energy intake to inc_i2, 
				# which we do not want in these simulations
				
				pars['it1']=0
				pars['it2']=np.inf
				
				pars_l=list(pars.values()) 
				pars_npa=np.array(pars_l) 
				pars_n=list(pars.keys()) 
				
				
				sol=sivp(odde,[0,300],y0,method='LSODA',args=[pars_npa])
				
				y0=sol.y[:,-1]
				# getting new initial condition for actual simulations
				
				#setting parameters back to their original values.
				pars['tau_b']=tau_b_or
				pars['tau_w']=tau_w_or
				pars['inc_i1']=inc_i1
				
				
				pars_l=list(pars.values()) 
				pars_npa=np.array(pars_l) 
				pars_n=list(pars.keys()) 
				
				tspan=[0,2700]
				
				# Generating solution trajectories, where terminal event f=unction determines 
				# whether the solution trajectory reaches to the threshold for imposed calorie restriction, 
				# indicating that the remission is possible for the initial condition for the given calorie restriction.
				sol=sivp(odde,tspan,y0,method='LSODA',args=[pars_npa],events=eventf)
				
				bbmi=sol.y[-1]/(pars['height']**2)
				bb=sol.y[4]
				
				ars1=np.where(bmius<=bbmi[-1])
				
				sols[i,j]=sol
				
				if bbus[ars1[0][0]]<bb[-1]:
					cols[i,j]=1
				else:
					cols[i,j]=0
		
		'''
		The following code block uses solution trajectories and cols to create the color map and color bar.
		It uses pyplot.fill function. The boundries of the region to be filled is determined by following;
		on the left,the branch of unstable steady states 
		on the right, the maximum BMI value
		on the top and the bottom, solution trajectories, or relevant axis limit.
		The color brightness is inversely proportial to the calorie restriction necessary for remission.
		'''
		inc_i1s=np.array(inc_i1s)
		inc_i1s=inc_i1s[cols.any(axis=0)]
		cols=cols[:,cols.any(axis=0)]
		
		ncols=len(inc_i1s)
		
		cbounds=[1,0.7,0.6,0.5,0.4,0.3,0.2,0.12,0]
		
		norm = colors.Normalize(vmin=0, vmax=1)
		cmap = cm.ScalarMappable(norm=norm, cmap=cm.YlGnBu)
		cmap.set_array([])
		
		
		tbmi1=sols[:,0][cols[:,0]==1][0].y[-1]/1.8**2
		tb1=sols[:,0][cols[:,0]==1][0].y[4]
		
		tb2=np.linspace(np.min(tb1),f2(25),10)
		tbmi2=f(tb2)
		
		poly=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		
		poly[0]=ax.fill(np.array([0,50,50,0]),np.array([0,0,2000,2000]),c=cmap.to_rgba(0))
		
		poly[1]=ax.fill(np.concatenate([tbmi1,tbmi2,np.array([25,25,50,50])]),
		  np.concatenate([tb1,tb2,np.array([f(25),0,0,np.max(tb1)])]),c=cmap.to_rgba(1))
		
		for j in range(len(inc_i1s)-1):
			
			tbmi1=sols[:,j][cols[:,j]==1][0].y[-1]/1.8**2
			tb1=sols[:,j][cols[:,j]==1][0].y[4]
			
			tbmi2=sols[:,j+1][cols[:,j+1]==1][0].y[-1]/1.8**2
			tb2=sols[:,j+1][cols[:,j+1]==1][0].y[4]
			
			tb2n=np.concatenate([tb2,np.linspace(np.min(tb2),np.min(tb1),10)])
			tbmi2n=np.concatenate([tbmi2,f(np.linspace(np.min(tb2),np.min(tb1),10))])
			
			poly[1+j]=ax.fill(np.append(tbmi1, tbmi2n[::-1]), np.append(tb1, tb2n[::-1]),c=cmap.to_rgba(0.85-inc_i1s[j]))
		
		j=j+1
		tbmi1=sols[:,j][cols[:,j]==1][0].y[-1]/1.8**2
		tb1=sols[:,j][cols[:,j]==1][0].y[4]
		
		tb2=np.linspace(np.min(tb1),np.max(bbus),20)
		tbmi2=f(tb2)
		
		poly[-1]=ax.fill(np.concatenate([tbmi1,tbmi2,np.array([np.max(bmius),np.max(bmius),50,50])]),
		  np.concatenate([tb1,tb2,np.array([np.max(bbus),2000,2000,np.max(bbus)])]),c=cmap.to_rgba(0.85-inc_i1s[-1]))
		
		ticks=np.concatenate([[0],0.85-inc_i1s,[1]])
		tlbls=[int(i*100+100) for i in inc_i1s]
		tlbls.insert(0, 'NR')
		tlbls.append('F')
		cbar = fig.colorbar(cmap, ticks=ticks, ax=ax,location='right',pad=0.01)
		cbar.ax.set_yticklabels(tlbls,size='medium')
		cbar.ax.set_ylabel('Max. DE$_i$ for remission (% BL)',labelpad=5.5)
		cbar.ax.invert_yaxis()
		plt.show()


def outputs(fn=12,bif_name='bifurcation_data.dat',it1=300,inc_i1s=[0.7,0.75],inc_i2s=[0.5,0.2],it2s=[70*30,80*30],dec_b=False,labels=None):
	from matplotlib.gridspec import GridSpec
	
	if fn==12:
		fignum=12
		pars['it1']=it1
		inc_i1s=[0.5,0.75, 0.75]
		inc_i2s=[0,0,0]
		it2s=[np.inf,np.inf,70*30]
		labels=['$DE_i$=150% BL','$DE_i$=175% BL','$DE_i$=175% BL -> 100% BL at T=70 months']
		dec_b=False
	elif fn==13:
		fignum=13
		pars['it1']=it1
		inc_i1s=[ 0.75, 0.75, 0.75]
		inc_i2s=[0.2,0.3,0]
		it2s=[70*30,70*30,88*30]
		labels=['$DE_i$=175% BL -> 120% BL at T=70 months','$DE_i$=175% BL -> 130% BL at T=70 months','$DE_i$=175% BL -> 100% BL at T=88 months']
		dec_b=True
	elif fn=='custom':
		fignum=14
		pars['it1']=it1
		inc_i1s=inc_i1s
		inc_i2s=inc_i2s
		it2s=it2s
		dec_b=dec_b
		if labels:
			labels=labels
		else:
			labels=[str(i+1) for i in range(len(inc_i1s))]
	
	
	gs=GridSpec(ncols=3, nrows=3, wspace=0.5,hspace=0.5, height_ratios=[1,1,1])
	
	axs=list(np.zeros(4))
	fig=plt.figure(fignum,tight_layout=False)
	axs[0]=fig.add_subplot(gs[0,0])
	axs[1]=fig.add_subplot(gs[0,1])
	axs[2]=fig.add_subplot(gs[0,2])
	axs[3]=fig.add_subplot(gs[1:,:])
	
	bif_fig(bif_name=bif_name,pars=pars,fig=fig,ax=axs[3],dec_bound=dec_b,ms=100)
	
	pars['it1']=300
	pars['tau_w']=1
	
	info=read_info('model.ode.pars')
	
	for i,(inc_i1,it2) in enumerate(zip(inc_i1s,it2s)):
		pars['inc_i2']=inc_i2s[i]
		pars['inc_i1']=inc_i1
		pars['it2']=it2
		
		sol=simul(odefun=odde, tspan=[0,4800], y0=y0, pars=pars, info=info,draw_fig=False)
		
		axs[0].plot(sol.t/30,sol.y[7]/pars['height']**2,label=labels[i])
		axs[0].set_xlim(0,144)
		axs[0].set_ylabel(r'BMI ($kg/m^2$)',size='medium')
		axs[0].set_xlabel('Time (months)',size='medium')
		
		axs[1].plot(sol.t/30,sol.y[4])
		axs[1].set_xlim(0, 144)
		axs[1].set_ylabel(r'$\beta$ (mg)',size='medium')
		axs[1].set_xlabel('Time (months)',size='medium')
		
		axs[2].plot(sol.t/30,sol.y[0])
		axs[2].set_xlim(0, 144)
		axs[2].set_ylabel('Glucose (mg/dl)',size='medium')
		axs[2].set_xlabel('Time (months)',size='medium')
		axs[2].hlines(y=[100,125],xmin=0,xmax=144,colors=['k','r'],ls=':',lw=0.8)
		
		axs[3].plot(sol.y[7]/pars['height']**2,sol.y[4])
		
	axs[0].legend(fontsize='medium',framealpha=0.5,loc=(0,1.15))
	
	fig.subplots_adjust(top=0.83,
	bottom=0.075,
	left=0.125,
	right=0.9,
	hspace=0.2,
	wspace=0.2)
	fig.set_size_inches([7.8,8.4])
	
	import matplotlib.transforms as mtransforms
	import string
	letters=string.ascii_uppercase
	trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
	
	for i,ax in enumerate(axs):
	 	ax.text(-0.1, 1.05, letters[i] , transform=ax.transAxes ,fontsize='medium', weight='bold',va='bottom', fontfamily='arial')
	 	ax.tick_params(axis='both', which='major', labelsize=10)
	fig.show()

def animation(fignum=123,bif_name='bifurcation_data.dat',it1=300,it2=2100,inc_i1=0.75,inc_i2=0.1,t0=0,tend=3600,dec_bound=False):
	pars['it1']=it1
	pars['inc_i1']=inc_i1
	pars['it2']=it2
	pars['inc_i2']=inc_i2
	
	tem=tend/30
	
	info=read_info('model.ode.pars')
	gs=GridSpec(ncols=3, nrows=3, wspace=0.5,hspace=0.5, height_ratios=[1,1,1])
	
	axs=list(np.zeros(4))
	fig=plt.figure(fignum,tight_layout=False)
	axs[0]=fig.add_subplot(gs[0,0])
	axs[1]=fig.add_subplot(gs[0,1])
	axs[2]=fig.add_subplot(gs[0,2])
	axs[3]=fig.add_subplot(gs[1:,:])
	
	parso=pars.copy()
	bif_fig(bif_name=bif_name,pars=pars,fig=fig,ax=axs[3],dec_bound=dec_bound)
	
	
	sol=simul(odefun=odde, tspan=[t0,tend], y0=y0, pars=parso,draw_fig=False)
	
	axs[0].plot(sol.t/30,sol.y[7]/pars['height']**2,lw=0.5)
	axs[0].set_xlim(0,tem)
	axs[0].set_ylabel(r'BMI ($kg/m^2$)',size='medium')
	axs[0].set_xlabel('Time (months)',size='medium')
	
	axs[1].plot(sol.t/30,sol.y[4],lw=0.5)
	axs[1].set_xlim(0, tem)
	axs[1].set_ylabel(r'$\beta$ (mg)',size='medium')
	axs[1].set_xlabel('Time (months)',size='medium')
	
	axs[2].plot(sol.t/30,sol.y[0],lw=0.5)
	axs[2].set_xlim(0, tem)
	axs[2].set_ylabel('Glucose (mg/dl)',size='medium')
	axs[2].set_xlabel('Time (months)',size='medium')
	axs[2].hlines(y=[100,125],xmin=0,xmax=144,colors=['k','r'],ls=':',lw=0.8)
	
	axs[3].plot(sol.y[7]/pars['height']**2,sol.y[4],lw=0.5)
	
	fig.subplots_adjust(top=0.9,
	bottom=0.1,
	left=0.125,
	right=0.9,
	hspace=0.2,
	wspace=0.2)
	fig.set_size_inches([7.8,7.4])
	
	
	import matplotlib.transforms as mtransforms
	import string
	letters=string.ascii_uppercase
	trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
	
	for i,ax in enumerate(axs):
		ax.text(-0.1, 1.05, letters[i] , transform=ax.transAxes ,fontsize='medium', weight='bold',va='bottom', fontfamily='arial')
		ax.tick_params(axis='both', which='major', labelsize=10)
	
	bmi=sol.y[-1]/pars['height']**2
	b=sol.y[4]
	g=sol.y[0]
	ts=sol.t
	
	t=np.linspace(t0,tend,100)
	
	bmi=interp1d(ts,bmi)(t)
	b=interp1d(ts,b)(t)
	g=interp1d(ts,g)(t)
	
	
	ax0=axs[0]
	ax1=axs[1]
	ax2=axs[2]
	ax3=axs[3]
	
	xdata, ydata = [], []
	ln0, = ax0.plot([], [], 'r-', marker='o')
	ln1, = ax1.plot([], [], 'r-', marker='o')
	ln2, = ax2.plot([], [], 'r-', marker='o')
	ln3, = ax3.plot([], [], 'r-', marker='o')
	
	
	nc=100
	
	gl=np.linspace(min(g),max(g),nc)
	colorss = plt.cm.RdYlGn(np.linspace(1,0,nc))
	
	if not dec_bound:
		tlbls=np.arange(round(g.min()),round(g.max()),10)
		norm = colors.Normalize(vmin=0, vmax=1)
		cmap = cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn_r)
		cmap.set_array([])
		cbar = fig.colorbar(cmap, ax=ax3,location='right',pad=0.01)
		cbar.ax.set_yticklabels(tlbls,size='medium')
		cbar.ax.set_ylabel('Glucose mg/dl',labelpad=5.5)
		cbar.ax.invert_yaxis()
	
	def update(i):
		cli=colorss[g[i]<=gl][0]
		ln3.set_data(bmi[i], b[i])
		ln3.set_markersize(8*(bmi[i]/25)**2)
		ln3.set_markerfacecolor(cli)
		ln3.set_markeredgecolor('k')
		
		ln0.set_data(t[i]/30, bmi[i])
		ln0.set_markersize(8)
		ln0.set_markerfacecolor(cli)
		ln0.set_markeredgecolor('k')
		
		ln1.set_data(t[i]/30, b[i])
		ln1.set_markersize(8)
		ln1.set_markerfacecolor(cli)
		ln1.set_markeredgecolor('k')
		
		ln2.set_data(t[i]/30, g[i])
		ln2.set_markersize(8)
		ln2.set_markerfacecolor(cli)
		ln2.set_markeredgecolor('k')
		
		return ln3,ln0,ln1,ln2
	
	# animate the plot
	ani = FuncAnimation(fig, 
						update, 
						frames=len(b), 
						interval=30, # delay of each frame in miliseconds
						blit=True,
						save_count=100)
	fig.show()
	
	return ani
