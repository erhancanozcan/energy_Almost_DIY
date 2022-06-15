import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
os.chdir("/Users/can/Desktop/logs")


#%%

objects_list = []
names=["50_96_0.35_100.0_10.0_0.01_060922_190841",
       "50_96_0.35_200.0_10.0_0.01_060922_190838",
       "50_96_0.35_300.0_10.0_0.01_060922_170946_gurobi"]
for name in names:
    with (open(name, "rb")) as openfile:
        while True:
            try:
                objects_list.append(pickle.load(openfile))
            except EOFError:
                break
#objects=objects[0]
#%%
prices=[]

for objects in objects_list:
    prices.append(objects['opt_price'])

#%%
objects=objects_list[1]

init_price=objects['price_lb']
opt_price=objects['opt_price']

real=objects['real']
dev=objects['dev']
real_before_optimizing_price=objects['real_before_changing_price']
dev_before_optimizing_price=objects['dev_before_changing_price']
gurobi_objective=objects['c_a_obj_list']
gurobi_home_epsilons=objects['home_weak_duality_epsilon']

opt_gap=objects['optimality_gap']
Q=objects['Q']
#%%
objects=objects_list[1]


slp_linearized_objective=objects['c_a_obj_list']
slp_linearized_home_epsilons=objects['home_linearized_weak_duality_epsilon']
slp_price=objects['opt_price']

slp_real_objective=objects['c_a_real_obj_list']
slp_real_home_epsilon=objects['home_real_weak_duality_epsilon']


real=objects['real']
dev=objects['dev']


objects['c_a_obj_list']



#%%

def price_plot(init_price,opt_price,y_lim=1):
    horizon=len(init_price)
    
    fig, ax = plt.subplots()
    ax.step(np.arange(horizon),opt_price,label="Coordination Agent' s Price",alpha=0.7)
    ax.step(np.arange(horizon),init_price,label="Random Price")
    #ax.legend(loc="upper right",["real power","desirable power"])
    ax.legend(loc="upper right")
    ax.set_title("Price Comparison per Interval")
    ax.set_ylabel("Dollar per Kwh")
    ax.set_ylim((-1,y_lim))#y_label_helper counts the number of homes sent to the function.
    ax.set_xlabel("Time Index")
    fig
    

def power_plot(real,dev,plot_type="total",y_lim=15):
    """
        plot_type : wm,
                    oven,
                    dryer,
                    hvac,
                    ewh,
                    ev,
                    total
        
        y_lim :     Useful parameter to adjust the length of y axis.
    """
    s_h_real=0
    s_h_dev=0
    title_helper="Single Home "
    y_label_helper=1
    
    if plot_type!="total":
        #If input is a list, total power consumed by the community is calculated.
        if type(real) is list:
            title_helper="Community "
            for (h_real,h_dev) in zip(real,dev):
                y_label_helper+=1
                s_h_real+=h_real[plot_type]
                s_h_dev+=h_dev[plot_type]
            s_h_des=s_h_real-s_h_dev
        else:
            #If input is not a list, power consumed by a single is calculated.
            s_h_real=real[plot_type]
            s_h_des=s_h_real-dev[plot_type]
    else:
        if type(real) is list:
            #If input is a list, total power consumed by the community is calculated.
            title_helper="Community "
            for (h_real,h_dev) in zip(real,dev):
                y_label_helper+=1
                for key in list(h_real.keys()):
                    if key != 'pv' and key != 'refrigerator':
                        s_h_real+=h_real[key]
                        s_h_dev+=h_dev[key]
            s_h_des=s_h_real-s_h_dev
        else:
            #If input is not a list, power consumed by a single is calculated.
            h_real=real
            h_dev=dev
            for key in list(h_real.keys()):
                if key != 'pv' and key != 'refrigerator' :
                    s_h_real+=h_real[key]
                    s_h_dev+=h_dev[key]
            s_h_des=s_h_real-s_h_dev
            
        
    horizon=len(s_h_real)
    
    fig, ax = plt.subplots()
    ax.step(np.arange(horizon),s_h_real,label="optimal power")
    ax.step(np.arange(horizon),s_h_des,label="desirable power",alpha=0.7)
    #ax.legend(loc="upper right",["real power","desirable power"])
    ax.legend(loc="upper right")
    ax.set_title("Load Comparison per Interval  ("+title_helper+plot_type.upper()+" Power)")
    ax.set_ylabel("Kw")
    ax.set_ylim((-2,y_lim*y_label_helper))#y_label_helper counts the number of homes sent to the function.
    ax.set_xlabel("Time Index")
    fig
#%%
#i denotes the home_index
i=10

h_real=real[i]
h_dev=dev[i] 
power_plot(h_real,h_dev,plot_type="hvac",y_lim=10)
#%%
power_plot(real,dev,plot_type="total",y_lim=12)
#%%
power_plot(real_before_optimizing_price,dev_before_optimizing_price,plot_type="total",y_lim=12)
#%%
price_plot(init_price,opt_price,y_lim=1.5)
#%%

price_plot(init_price,slp_price,y_lim=2)   
    
#%%
array([1.36118559, 1.29115719, 1.3746455 , 1.35863538, 1.22321618,
       1.43254307, 1.31792451, 1.44137649, 1.18690465, 1.45506178,
       1.32353471, 1.30810529, 1.21768971, 1.26991059, 1.38435566,
       1.36561728, 1.34937889, 1.32912691, 1.45396535, 1.50063381,
       1.19976603, 1.31400708, 1.39609077, 1.36384483, 1.1843656 ,
       1.31114408, 1.2810062 , 1.29202202, 1.44034244, 1.30325398,
       1.28640294, 1.43895554, 1.23516816, 1.52258   , 1.15564643,
       1.34906371, 1.22481862, 1.46376446, 1.23072133, 1.3749114 ,
       1.39785298, 1.32529359, 1.3317275 , 1.37439991, 1.4561245 ,
       1.208719  , 1.2524867 , 1.37303969, 1.54320167, 1.24105498,
       1.29570542, 1.42624431, 1.32861109, 1.27868144, 1.17624523,
       1.15426756, 1.1434547 , 1.28779643, 1.24860686, 1.37916826,
       1.29318594, 1.30528849, 1.36425922, 1.22411209, 1.41235343,
       1.33316626, 1.33692742, 1.43309337, 1.19423028, 1.2885457 ,
       1.50127047, 1.47379191, 1.48997124, 1.40624053, 1.33504655,
       1.25442025, 1.32178564, 1.17825082, 1.29568518, 1.16573919,
       1.41646341, 1.45654391, 1.43291142, 1.43220502, 1.38786353,
       1.23889801, 1.41782374, 1.14800246, 1.12105304, 1.38236873,
       1.35251132, 1.17338217, 1.25667571, 1.34369761, 1.17599018,
       1.26282287])

#%%




# def home_power_plot(home,real_power,price,plot_type="total",plot=True):
    
#     horizon=len(home.hvac_desirable_load)
#     if plot_type=="total":
#         desirable_power=home.ewh_desirable_load+home.ev_desirable_load+\
#             home.hvac_desirable_load+home.refrigerator_desirable_load+\
#             home.oven_desirable_load+home.wm_desirable_load+\
#             home.dryer_desirable_load
            
            
#         real_power_tmp=real_power['ewh']+real_power['ev']+\
#                          real_power['hvac']+real_power['refrigerator']+\
#                          real_power['oven']+real_power['wm']+\
#                          real_power['dryer']
#     else:
#         real_power_tmp=copy.deepcopy(real_power[plot_type])
#         if plot_type=="ewh":
#             desirable_power=home.ewh_desirable_load
#         elif plot_type=="ev":
#             desirable_power=home.ev_desirable_load
#         elif plot_type=="hvac":
#             desirable_power=home.hvac_desirable_load
#         elif plot_type=="refrigerator":
#             desirable_power=home.refrigerator_desirable_load
#         elif plot_type=="oven":
#             desirable_power=home.oven_desirable_load
#         elif plot_type=="wm":
#             desirable_power=home.wm_desirable_load
#         elif plot_type=="dryer":
#             desirable_power=home.dryer_desirable_load

        
        
        
                         
#     if plot==True:
        
#         fig, ax = plt.subplots()
#         ax.step(np.arange(horizon),real_power_tmp,label="real power")
#         ax.step(np.arange(horizon),desirable_power,label="desirable power",alpha=0.7)
#         #ax.legend(loc="upper right",["real power","desirable power"])
#         ax.legend(loc="upper right")
#         ax.set_title("Load Comparison per Interval  ("+plot_type.upper()+")")
#         ax.set_ylabel("Kw")
#         ax.set_ylim((0,12))
#         ax.set_xlabel("Time Index")
#         fig
            
#     #note that price is in terms of kwh.
#     fee_desirable_load=np.dot(desirable_power,price)*24/horizon#in terms of Kwh.
#     fee_real_load= np.dot(real_power_tmp,price)*24/horizon#in terms of Kwh.
    
#     real_power_tmp=np.sum(real_power_tmp)*24/horizon#in terms of Kwh.
#     desirable_power=np.sum(desirable_power)*24/horizon#in terms of Kwh.
    
#     print("Daily fee ("+plot_type+ ") desirable load: %.2f" %fee_desirable_load)
#     print("Daily fee ("+plot_type+ ") real load: %.2f" %fee_real_load)
#     print(plot_type+ " desirable load: %.2f" %desirable_power)
#     print(plot_type+ " real load: %.2f" %real_power_tmp)
    
#     return fee_desirable_load,fee_real_load,real_power_tmp,desirable_power

#%%

objects_list = []
names=["10_96_0.35_300.0_10.0_0.01_052422_081558",
       "20_96_0.35_300.0_10.0_0.01_052422_082133",
       "30_96_0.35_300.0_10.0_0.01_052422_082854",
       "40_96_0.35_300.0_10.0_0.01_052422_083925",
       "50_96_0.35_300.0_10.0_0.01_052422_090708"]
for name in names:
    with (open(name, "rb")) as openfile:
        while True:
            try:
                objects_list.append(pickle.load(openfile))
            except EOFError:
                break

#%%
objects_list[0].keys()

time=[obj['optimization_time'].seconds for obj in objects_list ]
gap_p=[obj['optimality_gap'] for obj in objects_list ]

num_homes=[10,20,30,40,50]
#%%

fig, ax = plt.subplots()
ax.plot(num_homes,time,marker='o',label="Time to Reach 1% Optimality Gap")
#ax.plot(eps,no_success_new_env_old_opt_policy,label="Old Policy in New Environment",alpha=0.7)
#ax.plot(eps,no_success_new_env_robust_policy,label="Robust Policy in New Environment",marker='o',alpha=0.7)
ax.legend(loc="upper right")
ax.set_title("Elapsed Time vs Homes")
ax.set_ylabel("Seconds")
#ax.set_ylim((1900,5000))#y_label_helper counts the number of homes sent to the function.
#ax.set_xlim((-0.001,0.003))#y_label_helper counts the number of homes sent to the function.
ax.set_xlabel("The Number of Homes in Community")
fig
#%%
fig, ax = plt.subplots()
ax.plot(num_homes,gap_p,marker='o',label="Optimality Gap Percentage")
#ax.plot(eps,no_success_new_env_old_opt_policy,label="Old Policy in New Environment",alpha=0.7)
#ax.plot(eps,no_success_new_env_robust_policy,label="Robust Policy in New Environment",marker='o',alpha=0.7)
ax.legend(loc="upper right")
ax.set_title("Optimality Gap Percentage vs Homes")
ax.set_ylabel("Optimality Gap %")
#ax.set_ylim((1900,5000))#y_label_helper counts the number of homes sent to the function.
#ax.set_xlim((-0.001,0.003))#y_label_helper counts the number of homes sent to the function.
ax.set_xlabel("The Number of Homes in Community")
fig



