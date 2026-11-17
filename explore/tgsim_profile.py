import pandas as pd, numpy as np
d=pd.read_csv('data_raw/tgsim_i9094s/I90_94_stationary_final.csv')
d.to_parquet('data_raw/tgsim_i9094s/all.parquet')
print(d.shape); print(d.dtypes)
for r,g in d.groupby('run_index'):
    print('run',r,'veh',g.id.nunique(),'T',g.time.min(),g.time.max(),'x',g.xloc_kf.min().round(),g.xloc_kf.max().round(),'y',g.yloc_kf.min().round(),g.yloc_kf.max().round(),
      'lanes',sorted(g.lane_kf.unique()),'av',g[g.av!='no'].id.unique()[:5])
print(d.type_most_common.value_counts())
d=d.sort_values(['run_index','id','time'])
d['pl']=d.groupby(['run_index','id']).lane_kf.shift()
lc=d[(d.pl.notna())&(d.pl!=d.lane_kf)]
print(lc.groupby(['run_index','pl','lane_kf']).size().to_string())
