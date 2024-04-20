
import numpy as np
#numpy.random._bit_generator = numpy.random.bit_generator
import agent
import matplotlib.pyplot as plt
import seaborn as sn
from ohlc import ohlc
from portfolio import portfolio
from HierarchicalRiskParity import  HierarchicalRiskParity
import frontier

ohlc_obj = ohlc()
ohlc_obj.load()
portf_obj = portfolio(ohlc=ohlc_obj.ohlc)
td_equity_dict = agent.getTdaPosition(type='EQUITY')
fidelity_equity_dict = agent.getFidelityPosition(type='EQUITY')
td_option_dict = agent.getTdaPosition(type='OPTION')
fidelity_option_dict = agent.getFidelityPosition(type='OPTION')

#Merge
equity_dict = {**td_equity_dict, **fidelity_equity_dict} 
option_dict = {**td_option_dict, **fidelity_option_dict} 

for symbol,pos in  equity_dict.items():
    portf_obj.update_pos(symbol, pos)
    
print (portf_obj.sigma(method='hist') )    
print (portf_obj.sigma(method='garch') )

corr = portf_obj.corr()
heatmap = sn.heatmap(np.round(corr, 2), vmin=-1, vmax=1, annot=True, cmap='viridis')
plt.show()
hrp =  HierarchicalRiskParity(portf_obj.corr(), date=ohlc_obj.current_date)


# Perform clustering
hrp.perform_clustering()
heatmap = sn.heatmap(np.round(hrp.linkage_matrix, 2), vmin=-1, vmax=1, annot=True, cmap='viridis')
#plt.show()
# Allocate weights
hrp.allocate_weights()
print (hrp.weights)
# Plot the dendrogram
hrp.plot_dendrogram()

    
#plot_points(portf.mu().values, portf.sigma().values, portf.mu().index.to_list())
frontier.plot_points(np.append(portf_obj.mu().values, portf_obj.mu_p()), np.append(portf_obj.sigma().values, portf_obj.sigma_p()),
            portf_obj.mu().index.to_list() + ['Current portfolio'], ohlc_obj.current_date)
frontier.plot_min_var_frontier(portf_obj.mu().values, portf_obj.cov().values)
rf = 0.05
frontier.plot_Capital_Allocation_Line(rf, portf_obj.mu().values, portf_obj.cov().values)
#plt.show()

portf_obj.opt_min_risk()
portf_obj.opt_max_mean_with_sigma()
sigma = portf_obj.sigma_p()
print (portf_obj)