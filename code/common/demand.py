import numpy as np
import pandas as pd
import random


def initialize_demand(home,time_res=15):
    horizon=24*int(60/time_res)
    
    
    
    #water demand
    """
    TODO
        need to check mean of water demand.
        create variations
    """
    #num_sample denotes demand points
    num_sample=np.random.randint(2,5)
    ind=np.random.choice(horizon-6,num_sample)
    tmp=np.random.normal(30,10,num_sample)
    tmp=abs(tmp)
    
    water_demand=np.zeros(horizon)
    water_demand[ind]=tmp#in terms of kg
    home.ewh_demand=water_demand

    #miles demand
    miles=np.random.randint(5,9)
    tmp=np.zeros(horizon)
    tmp[np.random.randint(30, 32):np.random.randint(33, 35)]=miles
    tmp[np.random.randint(66, 68):np.random.randint(69, 71)]=miles
    home.ev_demand=tmp
    
    
    #temperature demand
    tmprtr=np.random.randint(20,25)
    home.set_temperature=np.repeat(tmprtr,horizon)
    home.deadband=1
    
    #other appliances
    
    home.refrigerator_demand=np.repeat(1,horizon)
    ##
    s=np.random.randint(60, 80)
    tmp=np.zeros(horizon)
    tmp[s:s+4]=1
    home.oven_demand=tmp
    ##
    
    tmp=np.zeros(horizon)
    s=np.random.randint(0, 90)
    tmp=np.zeros(horizon)
    tmp[s:s+4]=1
    home.wm_demand=tmp
    ##
    
    tmp=np.zeros(horizon)
    s=np.random.randint(0, 90)
    tmp=np.zeros(horizon)
    tmp[s:s+4]=1
    home.dryer_demand=tmp
    
    
    return home
    #raise NotImplemented