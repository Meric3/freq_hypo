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





# +
import pandas as pd
import numpy as np

df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # 임상 정보
df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # 트랙 목록
df_labs = pd.read_csv('https://api.vitaldb.net/labs')  # 검사 결과
# -

df_cases

df_labs

caseids = list(
    set(df_trks[df_trks['tname'] == 'Solar8000/ART_MBP']['caseid']) & 
    set(df_cases[df_cases['department'] == 'General surgery']['caseid']) &
    set(df_cases[df_cases['emop'] == 1]['caseid'])
)
caseids = caseids[:100]
print('Total {} cases found'.format(len(caseids)))

# +
# Set blood pressure threshold
mbp_thresholds = np.arange(40, 80)

# Save the final result
df = pd.DataFrame()
for caseid in caseids:
    print('loading {}...'.format(caseid), flush=True, end='')

    # Column ['anend'] : anesthesia end time
    aneend = df_cases[(df_cases['caseid'] == caseid)]['aneend'].values[0]

    # Last creatinine concentration before surgery
    preop_cr = df_labs[(df_labs['caseid'] == caseid) & (df_labs['dt'] < 0) & (df_labs['name'] == 'cr')].sort_values(by=['dt'], axis=0, ascending=False)['result'].values.flatten()
    if len(preop_cr) == 0:
        print('no preop cr')
        continue
    preop_cr = preop_cr[0]

    # Maximum creatinine concentration within 48 hours after surgery
    postop_cr = df_labs[(df_labs['caseid'] == caseid) & (df_labs['dt'] > aneend) &
        (df_labs['dt'] < aneend + 48 * 3600) & (df_labs['name'] == 'cr')]['result'].max(skipna=True)
    if not postop_cr or np.isnan(postop_cr):
        print('no postop cr')
        continue

    # KDIGO stage I
    aki = postop_cr > preop_cr * 1.5

    # Blood pressure during surgery
    tid_mbp = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'Solar8000/ART_MBP')]['tid'].values[0]
    mbps = pd.read_csv('https://api.vitaldb.net/' + tid_mbp).values[:,1]
#     mbps = vitaldb.load_case(caseid, 'ART_MBP').flatten()
    mbps = mbps[~np.isnan(mbps)]
    mbps = mbps[(mbps > 20) & (mbps < 150)]
    if len(mbps) < 10:
        print('no mbp')
        continue

    # Calculate the percentage that stays for the time as increasing the blood pressure by 1 unit.
    row = {'aki':aki}
    for mbp_threshold in mbp_thresholds:
        row['under{}'.format(mbp_threshold)] = np.nanmean(mbps < mbp_threshold) * 100

    # Append the result into row
    df = df.append(row, ignore_index=True)

    print('{} -> {}, {}'.format(preop_cr, postop_cr, 'YYYYYEESSS AKI' if aki else 'no AKI'))

print('{} AKI {:.1f}%'.format(df['aki'].sum(), df['aki'].mean() * 100))
# -










