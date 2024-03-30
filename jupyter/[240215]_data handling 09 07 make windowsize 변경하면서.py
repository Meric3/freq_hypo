# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: tor19py37
#     language: python
#     name: tor19py37
# ---

# %reload_ext autoreload
# %autoreload 2

# +
import os
import pickle
import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

from datetime import timezone, timedelta, datetime
from pathlib import Path

import scipy.interpolate as ip
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt


# + code_folding=[1]
def load_trk(tid, interval=1):
    if isinstance(tid, list) or isinstance(tid, set) or isinstance(tid, tuple):
        return load_trks(tid, interval)
    try:
        url = 'https://api.vitaldb.net/' + tid
        dtvals = pd.read_csv(url, na_values='-nan(ind)').values
    except:
        return np.empty(0)

    if len(dtvals) == 0:
        return np.empty(0)
    
    dtvals[:,0] /= interval  # convert time to row
    nsamp = int(np.nanmax(dtvals[:,0])) + 1  # find maximum index (array length)
    ret = np.full(nsamp, np.nan)  # create a dense array
    

    if np.isnan(dtvals[:,0]).any():  # wave track
        if nsamp != len(dtvals):  # resample
            ret = np.take(dtvals[:,1], np.linspace(0, len(dtvals) - 1, nsamp).astype(np.int64))
            case_valid_mask = ~np.isnan(ret)
            ret = pd.DataFrame(ret).fillna(method='ffill').values.flatten()
            mbps = ret
            x0 = np.arange(0,mbps.shape[0],1)
            spl = splrep(x0, mbps)
            x1 = np.linspace(0, mbps.shape[0], mbps.shape[0]*1)
            y1 = splev(x1, spl)   
            ret = y1
            case_valid_mask = ~np.isnan(ret)
            ret = ret[(np.cumsum(case_valid_mask) != 0) & (np.cumsum(case_valid_mask[::-1])[::-1] != 0)]
            ret = pd.DataFrame(ret).fillna(method='ffill').values.flatten()
        
        else:
            ret = dtvals[:,1]
    else:  # numeric track
        for idx, val in dtvals:  # copy values
            ret[int(idx)] = val
        mbps = ret
        case_valid_mask = ~np.isnan(mbps)
        mbps = mbps[(np.cumsum(case_valid_mask) != 0) & (np.cumsum(case_valid_mask[::-1])[::-1] != 0)]
        mbps = pd.DataFrame(mbps).fillna(method='ffill').values.flatten()
        x0 = np.arange(0,mbps.shape[0],1)
        spl = splrep(x0, mbps)
        x1 = np.linspace(0, mbps.shape[0], mbps.shape[0]*100)
        y1 = splev(x1, spl)   
        ret = y1
    return ret


def load_trks(tids, interval=1):
    trks = []
    maxlen = 0
    for tid in tids:
        trk = load_trk(tid, interval)
        trks.append(trk)
        if len(trk) > maxlen:
            maxlen = len(trk)
    if maxlen == 0:
        return np.empty(0)
    ret = np.full((maxlen, len(tids)), np.nan)  # create a dense array
    for i in range(len(tids)):  # copy values
        ret[:len(trks[i]), i] = trks[i]
    for i in range(len(tids)):  # copy values
        case_valid_mask = ~np.isnan(ret[:, i])
        ret[:, i] = pd.DataFrame(ret[:, i]).fillna(method='ffill').values.flatten()
    return ret[100:,:]


# -

def load_case(tnames, caseid=None, interval=1):
    global dftrks

    if isinstance(caseid, list) or isinstance(caseid, set) or isinstance(caseid, tuple):
        return load_cases(tnames, caseid, interval, 9999)

    if not caseid:
        return None
    if dftrks is None:
        dftrks = pd.read_csv("https://api.vitaldb.net/trks")

    tids = []
    for tname in tnames:
        tid = dftrks[(dftrks['caseid'] == caseid) & (dftrks['tname'] == tname)]['tid'].values[0]
        tids.append(tid)
    
    return load_trks(tids, interval)


dftrks = pd.read_csv('https://api.vitaldb.net/trks')  # 트랙 목록
dfcases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보

# +
# caseids = list(
#     set(dftrks[dftrks['tname'] == 'Solar8000/ART_MBP']['caseid']) & \
#     set(dftrks[dftrks['tname'] == 'Solar8000/RR']['caseid']) & \
#     set(dftrks[dftrks['tname'] == 'Solar8000/PLETH_HR']['caseid']) & \
#     set(dftrks[dftrks['tname'] == 'Primus/ETCO2']['caseid']) 
# #     set(dfcases[dfcases['age'] > 18]['caseid'])
# #     set(dfcases[~dfcases['opname'].str.contains("transplant")]['caseid'])
# )
# print('Total {} cases found'.format(len(caseids)))
# np.random.shuffle(caseids)  # caseid를 무작위로 섞음
# -

caseids = list(
    set(dftrks[dftrks['tname'] == 'SNUADC/ART']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'SNUADC/ECG_II']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'SNUADC/PLETH']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'Primus/CO2']['caseid']) 
#     set(dfcases[dfcases['age'] > 18]['caseid'])
#     set(dfcases[~dfcases['opname'].str.contains("transplant")]['caseid'])
)
print('Total {} cases found'.format(len(caseids)))
np.random.shuffle(caseids)  # caseid를 무작위로 섞음

caseids = list(
    set(dftrks[dftrks['tname'] == 'Solar8000/ART_MBP']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'Solar8000/RR']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'Solar8000/PLETH_HR']['caseid']) & \
    set(dftrks[dftrks['tname'] == 'Primus/ETCO2']['caseid']) 
#     set(dfcases[dfcases['age'] > 18]['caseid'])
#     set(dfcases[~dfcases['opname'].str.contains("transplant")]['caseid'])
)
print('Total {} cases found'.format(len(caseids)))
np.random.shuffle(caseids)  # caseid를 무작위로 섞음

# +
# tnames = ['SNUADC/ART', 'SNUADC/ECG_II', 'SNUADC/PLETH', 'Primus/CO2']
# trace_name = [ [ 'SNUADC/ART', 'abp' ],[ 'SNUADC/ECG_II', 'ecg' ],[ 'SNUADC/PLETH', 'ple' ],[ 'Primus/CO2', 'co2' ] ]
# -

tnames = ['Solar8000/ART_MBP', 'Solar8000/RR', 'Solar8000/PLETH_HR', 'Primus/ETCO2']
trace_name = [ [ 'Solar8000/ART_MBP', 'abp' ],[ 'Solar8000/RR', 'ecg' ],[ 'Solar8000/PLETH_HR', 'ple' ],[ 'Primus/ETCO2', 'co2' ] ]

# +
sampling_rate = 100 # Resampling (Hz)
interval = 1
input_length = [ 30 ] # Input data length (sec)
pred_lag = [ 300, 600, 900 ] # Prediction lag for 300, 600, and 900 sec (5-, 10-, 15-min prediction) 
pred_threshold = { 'hypo': lambda x: x < 65, # Threshold for hypotension (mmHg)
                   'normo': lambda x: x >= 65 }
pred_min_length = { 'hypo': 60, # Minimum duration (sec) for defining hypotensive event
                    'normo': 1200 } # for non-hypotensive event (normal)
convert_dir = './converted/' # Input path
processed_dir = './processed/' # Output path

# converted_path = convert_dir+str(caseid)+'.pkl'
# processed_path = processed_dir+'{}.pkl'.format(caseid)
# -



caseid = 3568 # Case number
processed_path = processed_dir+'{}.pkl'.format(caseid)
tids = []
for tname in tnames:
    tid = dftrks[(dftrks['caseid'] == caseid) & (dftrks['tname'] == tname)]['tid'].values[0]
    print(tname)
    tids.append(tid)
case = load_trks(tids, interval)
print(case.shape)

# # tptp

input_length_ = [10,20,40,50,60 ]
for inputs_ in input_length_:
    input_length = [inputs_]

    processed_dir = '/hdd2/mjh319/data/hypotension/processed/'
    processed_dir_ = Path(processed_dir, str(inputs_))
    processed_dir_.mkdir(parents=True, exist_ok=True)
    processed_dir = processed_dir +  str(inputs_) + "/"

    tz = timezone(timedelta(hours=9))
    now = datetime.now(tz)
    mon = format(now.month, '02')
    day = format(now.day, '02')
    h = format(now.hour, '02')
    m = format(now.minute, '02')
    s = format(now.second, '02')
    today = mon+day+h+m+s

    processed_dir_ = Path(processed_dir, today)
    processed_dir = processed_dir + today + "/"
    processed_dir_.mkdir(parents=True, exist_ok=True)
    print(processed_dir)

    sampling_rate = 100 # Resampling (Hz)
    interval = 1
    # Input data length (sec)
    pred_lag = [ 100, 300, 600, 900, 1500 ] # Prediction lag for 300, 600, and 900 sec (5-, 10-, 15-min prediction) 
    pred_threshold = { 'hypo': lambda x: x < 65, # Threshold for hypotension (mmHg)
                       'normo': lambda x: x >= 65 }
    pred_min_length = { 'hypo': 60, # Minimum duration (sec) for defining hypotensive event
                        'normo': 1200 } # for non-hypotensive event (normal)


    peak_height = { 'abp':30, 'ecg':-999, 'ple': 10, 'co2':20 }
    peak_prominence = { 'abp':10, 'ecg':0.1, 'ple': 5, 'co2':10 }
    peak_distance = { 'abp':1.0, 'ecg':1.0, 'ple': 1.0, 'co2':1 }

    tolerance_interval = { 'abp':3, 'ecg':3, 'ple':3, 'co2':8 }
    tolerance_min = { 'abp':0, 'ecg':-1, 'ple':-10, 'co2':0 }
    tolerance_max = { 'abp':250, 'ecg':1, 'ple':100, 'co2':70 }

    # Prespecifications
    for index in caseids:   

        caseid = index # Case number
        processed_path = processed_dir+'{}.pkl'.format(caseid)
        tids = []
        for tname in tnames:
            tid = dftrks[(dftrks['caseid'] == caseid) & (dftrks['tname'] == tname)]['tid'].values[0]
            tids.append(tid)
        case = load_trks(tids, interval)
        print("case", caseid, "shape ", case.shape)

        source_ = {}
        source_end = 0
        j = 0 
        try:
            for full_name, track in trace_name:
                source_[track] = case[:,j]
                source_end = max ( source_end, case[:,j].shape[0])
                j += 1
        except:
            print("j error")
            continue

        valid = {}

        peaks = { }
        next_peaks = { }


        error_ = 0
        for track in source_.keys():
            valid[track] = np.ones ( source_end )
            valid[track][np.where ( source_[track] < tolerance_min[track] )] = 0
            valid[track][np.where ( source_[track] > tolerance_max[track] )] = 0


            peaks [ track ] = find_peaks ( source_[track])[0]
            if len(peaks [ track ] ) == 0:
                error_ = 1
                break

            next_peaks [ track ] = np.append ( peaks [ track ][ 1: ], source_end )
            invalid_peaks = np.where ( ( ( next_peaks[track] - peaks[track] ) <= tolerance_interval[track] * sampling_rate ) == False )[0]

            if track == 'abp':
                for x in invalid_peaks:
                    valid[track][ peaks[track][x] : next_peaks[track][x] ] = 0


        if error_ == 1:
            continue


        n_peaks = len ( peaks['abp'] )
        peak_map = np.zeros ( n_peaks-1 )


        for i in range ( n_peaks-1 ):
            peak_map[i] = np.mean ( source_['abp'][peaks['abp'][i]:peaks['abp'][i+1]])
            if peak_map[i] < 20 or peak_map[i] > 200: # Exclude rhythms with abnormal arterial pressure
                valid['abp'][ peaks['abp'][i] : next_peaks['abp'][i] ] = 0

        valid_peak_num = { }
        consecutive_peak_num = {}
        section_stack = {}

        for phase in [ 'hypo', 'normo' ]:
            # Extractt peaks based on hypotension threshold
            valid_peak_num[phase] = np.where ( pred_threshold[phase](peak_map) )[0]
            # Mark -1 for abnormal peaks
            for i, x in enumerate(valid_peak_num[phase]):
                if np.all (   valid['abp'][ peaks['abp'][x]   :   peaks['abp'][x+1]  ] ):
                    pass
                else:
                    valid_peak_num[phase][i] = -1

            # Exclude abnormal peaks
            valid_peak_num[phase] = valid_peak_num[phase][ valid_peak_num[phase] != -1 ]
            # Concatenate consecutive rhythms
            consecutive_peak_num[phase] = np.split(valid_peak_num[phase], np.where(np.diff(valid_peak_num[phase]) != 1)[0]+1)
            section_stack[phase] = []
            for i, x in enumerate(consecutive_peak_num[phase]):
                if len(x) > 1:
                    start = peaks['abp'][x[0]]
                    end = peaks['abp'][x[-1]+1]

                    if end - start >= pred_min_length[phase] * 5:
    #                     print("ASDASD")
                        section_stack[phase].append ( [start, end] )


        output = { 'input_length':[],
                    'abp':[],
                    'ecg':[],
                    'ple':[],
                    'co2':[],
                    'pred_lag':[],
                    'hypo':[],
                    'map':[] }

        for phase in [ 'hypo', 'normo' ]:
            for pred_start, pred_end in section_stack[ phase ]:
                for length in input_length:
                    for lag in pred_lag:
                        s_end = pred_start - lag * sampling_rate
                        if phase == 'hypo':
                            s_end = pred_start - lag * sampling_rate
                        else:
                            s_end = int ( ( pred_start + pred_end ) * 0.5 - ( lag * 0.5 * sampling_rate ) )
                        s_start = s_end - length * sampling_rate
                        for multi in range ( 2 ):
    #                         print("mul {} ".format(multi) )
                            if phase == 'hypo' and multi >= 1:
                                break
                            section_start = s_start + multi * 10
                            section_end = s_end + multi * 10
    #                         print("st {} sed{} sour{} ".format(section_start, section_end, source_end))
                            if section_start >= 0 and section_end <= source_end:
                                output['input_length'].append ( length )
                                for track in [ 'abp', 'ecg', 'ple', 'co2' ]:
                                    output[track].append ( source_[track][section_start:section_end] )
                                output['pred_lag'].append ( lag )
                                output['hypo'].append ( 1 if phase == 'hypo' else 0 )
                                output['map'].append ( np.mean ( source_['abp'][pred_start:pred_end] ) )


        print("{} {} {} {}".format(index, len(section_stack['hypo']), len(section_stack['normo']), len(output['abp'])  ))

        if len(output['abp']) < 3 or np.isnan(output['abp']).any() or \
        np.isnan(section_stack['hypo']).any() or np.isnan(section_stack['normo']).any():
            print("isnan eror")
        else:
            print("witing : ", processed_path)
            with open(processed_path, 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)









