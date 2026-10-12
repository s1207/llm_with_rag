"""Detect encoder pulse-timing irregularities and large voltage excursions.
Usage: python analyze_encoder.py DS0001.CSV --out encoder_results
Dependencies: numpy, pandas, scipy, matplotlib
The detected waveform valleys are *candidate encoder events*, not independently
verified edges. Adjust --prominence and --min-distance-us for your hardware.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze(csv_path, out, stride=100, prominence=2.5, min_distance_us=6.0,
            tolerance=0.20, dip_threshold=30.0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # Tektronix-style metadata followed by 'Waveform Data,' and timestamp,voltage lines.
    with open(csv_path, encoding='utf-8-sig', errors='replace') as f:
        for line_no, line in enumerate(f):
            if line.strip().startswith('Waveform Data'):
                skiprows = line_no + 1
                break
        else: raise ValueError('Cannot locate Waveform Data section')
    # Float32 minimizes RAM use for recordings containing millions of samples.
    raw = pd.read_csv(csv_path, skiprows=skiprows, header=None,
                      usecols=[0,1], names=['time_s','voltage_V'],
                      dtype={'time_s':'float64','voltage_V':'float32'})
    raw.dropna(inplace=True)
    n = len(raw)//stride
    t = raw.time_s.to_numpy()[:n*stride].reshape(n,stride).mean(axis=1)
    v = raw.voltage_V.to_numpy()[:n*stride].reshape(n,stride).mean(axis=1)
    dt = float(np.median(np.diff(t)))
    distance = max(1,round(min_distance_us*1e-6/dt))
    # Light smoothing protects against isolated sample noise.
    smooth = v
    valleys, props = find_peaks(-smooth, prominence=prominence,
                                distance=distance)
    # Exclude endpoints, which are susceptible to smoothing boundary effects.
    valleys=valleys[(valleys>5)&(valleys<len(v)-5)]
    edge_t=t[valleys]; intervals=np.diff(edge_t)*1e6
    typical=np.median(intervals)
    bad=np.flatnonzero(abs(intervals-typical)>tolerance*typical)
    events=[]
    for i in bad:
        events.append({'type':'pulse_interval','time_ms':edge_t[i+1]*1e3,
                       'start_ms':edge_t[i]*1e3,'end_ms':edge_t[i+1]*1e3,
                       'duration_us':intervals[i], 'expected_us':typical,
                       'deviation_percent':100*(intervals[i]/typical-1),
                       'minimum_V':np.nan})
    # Separate very deep low-voltage excursions from timing decisions.
    dip=v<dip_threshold
    ix=np.flatnonzero(np.diff(np.r_[False,dip,False]))
    for begin,end in ix.reshape(-1,2):
        if end-begin<3: continue
        events.append({'type':'deep_voltage_dip','time_ms':t[begin]*1e3,
                       'start_ms':t[begin]*1e3,'end_ms':t[end-1]*1e3,
                       'duration_us':(end-begin)*dt*1e6,'expected_us':np.nan,
                       'deviation_percent':np.nan,
                       'minimum_V':float(v[begin:end].min())})
    results=pd.DataFrame(events,columns=['type','time_ms','start_ms','end_ms','duration_us','expected_us','deviation_percent','minimum_V']).sort_values('time_ms')
    results.to_csv(out/'encoder_anomalies.csv',index=False)
    fig,axs=plt.subplots(3,1,figsize=(14,10),constrained_layout=True)
    axs[0].plot(t*1e3,v,lw=.5,color='#485870')
    for e in events:
        if e['type']=='deep_voltage_dip':
            axs[0].axvspan(e['start_ms'],e['end_ms'],color='#e69f00',alpha=.32)
    axs[0].set(ylabel='CH1 voltage (V)',title='Encoder waveform — orange: deep voltage dips')
    axs[1].plot(edge_t[1:]*1e3,intervals,lw=.8,color='#385f9a')
    axs[1].scatter(edge_t[1:][bad]*1e3,intervals[bad],color='#ce3333',s=25,zorder=4,label='Outlier')
    axs[1].axhline(typical,color='green',ls='--',label=f'Median {typical:.2f} µs')
    for level in [typical*(1-tolerance),typical*(1+tolerance)]: axs[1].axhline(level,color='gray',ls=':',lw=1)
    axs[1].set(ylabel='Valley-to-valley interval (µs)',title='Detected timing irregularities',xlabel='Time (ms)')
    axs[1].legend(loc='upper right')
    axs[2].hist(intervals,bins=75,color='#6b84a2')
    axs[2].axvline(typical,color='green',ls='--')
    axs[2].set(xlabel='Detected interval (µs)',ylabel='Count',title='Pulse-interval distribution')
    fig.savefig(out/'encoder_overview.png',dpi=175);plt.close(fig)
    # Zoom around each of the largest departures; do not show duplicate nearby windows.
    ranked=sorted(bad,key=lambda i:abs(intervals[i]-typical),reverse=True)
    chosen=[]
    for i in ranked:
        if all(abs(edge_t[i+1]-edge_t[j+1])>100e-6 for j in chosen): chosen.append(i)
        if len(chosen)>=6:break
    if chosen:
        fig,axes=plt.subplots(len(chosen),1,figsize=(13,3.0*len(chosen)),squeeze=False,constrained_layout=True)
        for axarr,i in zip(axes,chosen):
            ax=axarr[0]; center=edge_t[i+1]; left,right=center-50e-6,center+50e-6
            mask=(t>=left)&(t<=right)
            ax.plot(t[mask]*1e3,v[mask],lw=1,color='#345779')
            ax.axvspan(edge_t[i]*1e3,edge_t[i+1]*1e3,alpha=.20,color='red')
            ax.axvline(edge_t[i]*1e3,color='red',ls='--',lw=.8)
            ax.axvline(edge_t[i+1]*1e3,color='red',ls='--',lw=.8)
            ax.set(title=f'{center*1e3:.4f} ms | pulse interval {intervals[i]:.2f} µs vs {typical:.2f} µs',ylabel='Voltage (V)',xlabel='Time (ms)')
        fig.savefig(out/'encoder_anomaly_zooms.png',dpi=175);plt.close(fig)
    print('Typical valley interval:',round(typical,3),'µs')
    print('Intervals outside tolerance:',len(bad),'of',len(intervals))
    print('Deep voltage excursions:',sum(e['type']=='deep_voltage_dip' for e in events))
    print('Saved:',out.resolve())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('csv_path');ap.add_argument('--out',default='encoder_results');ap.add_argument('--stride',type=int,default=100)
    ap.add_argument('--prominence',type=float,default=2.5);ap.add_argument('--min-distance-us',type=float,default=6.)
    ap.add_argument('--tolerance',type=float,default=.20);ap.add_argument('--dip-threshold',type=float,default=30.)
    a=ap.parse_args();analyze(a.csv_path,a.out,a.stride,a.prominence,a.min_distance_us,a.tolerance,a.dip_threshold)
