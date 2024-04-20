import requests
from settings import *
from munch import DefaultMunch
import pandas as pd
import re
import os

TDA_OPTION_PATTERN = r'([a-zA-Z]+)_(\d+)([C|P])(\d+\.?\d*)'
FIDELITY_OPTION_PATTERN = r'[^a-zA-Z]*([a-zA-Z]*)(\d{6})([C|P]\d+\.?\d*)'

def auth():
    response = requests.post('https://api.tdameritrade.com/v1/oauth2/token', 
                            data={'grant_type':'refresh_token',
                                  'client_id':APIKEY,
                                  'refresh_token': refresh_token})
    if response.status_code == 200:
        access_token = response.json()['access_token']
        return access_token
    else:
        return ""

def td_request():
    global access_token
    failure=0
    while True:
        #print (access_token)
        response = requests.get(GET_POSITIONS_URI, 
                                headers={'Authorization': "Bearer "+access_token})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401 and failure < 4:
            access_token = auth()
            failure += 1
            continue
        else:
            raise Exception(response.status_code)
        
def getTdaPosition(type='EQUITY'):
    td_portf = td_request()
    equity_pos_dict = dict()
    option_pos_dict = dict()

    data_obj = DefaultMunch.fromDict(td_portf)
    td_positions=data_obj.securitiesAccount.positions
    for pos in td_positions:
        if pos.instrument.assetType == 'EQUITY':
            #portf.change_pos(pos.instrument.symbol, pos.longQuantity-pos.shortQuantity)
            equity_pos_dict[pos.instrument.symbol] = pos.longQuantity-pos.shortQuantity
        if pos.instrument.assetType == 'OPTION':
            option_pos_dict[pos.instrument.symbol] = pos.longQuantity-pos.shortQuantity
            
    if type == 'EQUITY':
        return equity_pos_dict
    elif type == 'OPTION':
        return option_pos_dict
    else:
        return {}
        
    
def getFidelityPosition(type='EQUITY'):
    equity_pos_dict = {}
    option_pos_dict = {}

    curr_dir = os.path.dirname(__file__)
    for f in [f"{curr_dir}/data/fidelity18.csv",f"{curr_dir}/data/fidelity20.csv"]:        
        df=pd.read_csv(f)
        for s in df['Symbol'].dropna().values:
            if not s or s == 'Pending Activity':
                continue
            m=re.compile(FIDELITY_OPTION_PATTERN).search(s)
            if m:
                o=m.group(1)+'_'+m.group(2)[2:]+m.group(2)[0:2]+ m.group(3)
                option_pos_dict[o] = option_pos_dict.get(o,0) + (df.loc[df['Symbol'] == s, 'Quantity'].values[0])
            else:
                if df.loc[df['Symbol'] == s, 'Account Name'].values[0].lower() == 'ROTH IRA'.lower() or \
                df.loc[df['Symbol'] == s, 'Account Name'].values[0].lower() == 'TRADITIONAL IRA'.lower():
                    equity_pos_dict[s] = equity_pos_dict.get(s,0) + (df.loc[df['Symbol'] == s, 'Quantity'].values[0])
                   
                   
    if type== 'EQUITY':
        return equity_pos_dict
    elif  type== 'OPTION':
        return option_pos_dict
    else:
        return {}