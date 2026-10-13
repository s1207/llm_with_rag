"""Analyze encoder pulses and intentional low-going LED-switch commands.
Usage: python analyze_encoder_full.py DS0001.CSV --out encoder_full_results
Requires numpy, pandas, scipy, matplotlib. Thresholds are configurable.
IMPORTANT: nominal 5V normalization uses the median high level of CH1; an
unknown scope/probe offset prevents absolute 0V/5V compliance assertions.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import maximum_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_csv(path):
    metadata = {}
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line_num, line in enumerate(f):
            if line.lstrip().startswith('Waveform Data'):
                break
            pieces=line.rstrip('\r\n').split(',')
            if len(pieces)>1: metadata[pieces[0]]=pieces[1]
        else: raise ValueError('Waveform Data section not found')
    df=pd.read_csv(path,skiprows=line_num+1,header=None,usecols=[0,1],names=['time_s','voltage_scope_V'],dtype={'time_s':'float64','voltage_scope_V':'float32'})
    return df.time_s.to_numpy(),df.voltage_scope_V.to_numpy(),metadata

def true_runs(mask):
    pos=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return pos.reshape(-1,2)

def analyze(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    t,v,meta=load_csv(args.csv); dt=float(np.median(np.diff(t[:10000])))
    high=float(np.median(v)); normalized=v*(5.0/high) # relative gain only; zero offset unverified
    # Coarse 100-ns blocks preserve fast transitions using max-min, not arithmetic averages.
    step=max(1,int(round(args.bin_ns*1e-9/dt))); N=len(v)//step
    blk=v[:N*step].reshape(N,step)
    lo=blk.min(axis=1);hi=blk.max(axis=1);tt=t[:N*step:step]
    env=maximum_filter1d(hi-lo,size=5)
    minimum_distance=max(1,int(args.min_separation_us*1e-6/(dt*step)))
    peak,props=find_peaks(env,prominence=args.prominence_scope_v,height=args.min_amplitude_scope_v,distance=minimum_distance)
    pulse_t=tt[peak]; interval=np.diff(pulse_t)*1e6
    normal=float(np.median(interval)); amplitudes=env[peak]
    widths=peak_widths(env,peak,rel_height=0.5)[0]*step*dt*1e6
    typical_amp=float(np.median(amplitudes)); typical_width=float(np.median(widths))
    # Deep switching events. Merge small interruptions/ringing within a single commanded drop.
    deep_threshold=high*args.deep_fraction
    raw=true_runs(lo<deep_threshold)
    groups=[]
    for start,end in raw:
        if groups and (start-groups[-1][1])*step*dt*1e6 <= args.merge_gap_us: groups[-1][1]=end
        else:groups.append([int(start),int(end)])
    switch=[]
    for start,end in groups:
        duration=(end-start)*step*dt*1e6
        if duration<args.min_switch_us:continue
        a=start*step;b=min(len(v),end*step)
        switch.append(dict(start_s=float(t[a]),end_s=float(t[b-1]),duration_us=float(duration),minimum_scope_V=float(v[a:b].min()),minimum_relative_5V=float(normalized[a:b].min())))
    switch_df=pd.DataFrame(switch)
    if len(switch_df)>1:
        periods=np.diff(switch_df.start_s.to_numpy())*1e6
        expected_switch=float(np.median(periods))
    else:periods=np.array([]);expected_switch=float('nan')
    anomalies=[]
    def add(kind,at,observed,expected,detail):
        anomalies.append({'type':kind,'time_ms':float(at*1000),'observed':float(observed),'expected':float(expected),'detail':detail})
    for j,gap in enumerate(interval):
        t_event=pulse_t[j+1]
        if gap>normal*(1+args.interval_tolerance):
            label='possible_missing_trigger' if gap>=normal*args.missing_multiple else 'long_trigger_interval'
            add(label,t_event,gap,normal,'Interval in microseconds; pulse absent vs obscured by commanded dip needs downstream measurement')
        elif gap<normal*(1-args.interval_tolerance):
            add('short_trigger_interval',t_event,gap,normal,'Interval in microseconds; possible extra pulse or ringing')
    for j,amp in enumerate(amplitudes):
        if amp<typical_amp*args.amplitude_fraction:
            add('reduced_transition_amplitude',pulse_t[j],amp,typical_amp,'Raw-scope-volts peak-to-peak envelope')
    for j,width in enumerate(widths):
        if width<typical_width*args.width_fraction:
            add('short_transition_signature',pulse_t[j],width,typical_width,'Envelope half-prominence width (µs), NOT digital TTL pulse width')
    for j,sw in switch_df.iterrows():
        if sw.minimum_relative_5V>args.low_max_relative_V:
            add('switch_low_not_near_zero',sw.start_s,sw.minimum_relative_5V,0.,'Relative 5V scale, baseline/offset unknown; cannot validate real 0V')
        if j and np.isfinite(expected_switch):
            gap=periods[j-1]
            if abs(gap-expected_switch)>expected_switch*args.switch_tolerance:
                add('irregular_switch_spacing',sw.start_s,gap,expected_switch,'Microseconds between intentional deep switching events')
    if len(switch_df):
        nominal_sw_width=float(switch_df.duration_us.median())
        for _,sw in switch_df.iterrows():
            if sw.duration_us<nominal_sw_width*args.switch_width_fraction:
                add('short_switch_low_duration',sw.start_s,sw.duration_us,nominal_sw_width,'µs under fixed deep-switch threshold')
            elif sw.duration_us>nominal_sw_width*(2-args.switch_width_fraction):
                add('long_switch_low_duration',sw.start_s,sw.duration_us,nominal_sw_width,'µs under fixed deep-switch threshold')
    # Fraction of time high window doesn't reach relative 5 V: compare local maximum (excluding commanded dip)
    # This is descriptive, NOT a hard fault absent calibrated zero and high reference.
    pd.DataFrame({'time_ms':pulse_t*1e3,'transition_amplitude_scope_V':amplitudes,'signature_width_us':widths}).to_csv(out/'detected_transitions.csv',index=False)
    switch_df.to_csv(out/'intentional_switches.csv',index=False)
    result=pd.DataFrame(anomalies,columns=['type','time_ms','observed','expected','detail']).sort_values('time_ms')
    result.to_csv(out/'anomalies.csv',index=False)
    fig,axs=plt.subplots(3,1,figsize=(15,10),constrained_layout=True)
    k=max(1,len(t)//80000)
    axs[0].plot(t[::k]*1e3,normalized[::k],lw=.55,color='#36587a')
    for sw in switch:axs[0].axvspan(sw['start_s']*1e3,sw['end_s']*1e3,color='orange',alpha=.2)
    axs[0].set(ylabel='Relative scaled voltage (V)',title='Whole waveform: orange = intentional low-going commands (not anomalies)')
    axs[1].plot(pulse_t[1:]*1e3,interval,lw=.8)
    abnormal=np.abs(interval-normal)>normal*args.interval_tolerance
    axs[1].scatter(pulse_t[1:][abnormal]*1e3,interval[abnormal],c='red',s=18,zorder=3)
    axs[1].axhline(normal,c='green',ls='--',label=f'median {normal:.2f} µs')
    axs[1].set(ylabel='Transition spacing (µs)',title='Fast trigger transition timing (red = outliers)');axs[1].legend()
    axs[2].plot(pulse_t*1e3,amplitudes,lw=.7,label='Transition signature amplitude (scope V)')
    axs[2].axhline(typical_amp*args.amplitude_fraction,c='red',ls='--',label='Low amplitude threshold')
    axs[2].set(ylabel='Peak-to-peak scope V',xlabel='Oscilloscope time (ms)',title='Envelope amplitude per transition');axs[2].legend()
    fig.savefig(out/'full_overview.png',dpi=160);plt.close(fig)
    # Detailed zoom around worst abnormal gaps (read original 1ns-resolution waveform).
    ranked=np.argsort(np.abs(interval-normal))[::-1]; chosen=[]
    for j in ranked:
        if abs(interval[j]-normal)<=normal*args.interval_tolerance:break
        if all(abs(pulse_t[j+1]-pulse_t[k+1])>80e-6 for k in chosen):chosen.append(int(j))
        if len(chosen)==8:break
    if chosen:
        fig,axes=plt.subplots(len(chosen),1,figsize=(15,2.7*len(chosen)),squeeze=False,constrained_layout=True)
        for (ax,),j in zip(axes,chosen):
            center=pulse_t[j+1];left=center-40e-6;right=center+30e-6
            a=np.searchsorted(t,left);b=np.searchsorted(t,right);stride=max(1,(b-a)//15000)
            ax.plot(t[a:b:stride]*1e3,normalized[a:b:stride],lw=.7)
            ax.axvspan(pulse_t[j]*1e3,pulse_t[j+1]*1e3,color='red',alpha=.18)
            ax.set(title=f'{center*1e3:.5f} ms: {interval[j]:.2f} µs vs {normal:.2f} µs',ylabel='Relative V',xlabel='Time (ms)')
        fig.savefig(out/'timing_anomaly_zooms.png',dpi=150);plt.close(fig)
    with open(out/'summary.txt','w') as f:
        f.write(f'Samples: {len(t):,}; sample interval: {dt*1e9:.3f} ns; duration: {(t[-1]-t[0])*1e3:.3f} ms\n')
        f.write(f'CH1 measured median high: {high:.3f} scope V; multiplied by {5/high:.6f} to set median to 5 relative V. ABSOLUTE 0V NOT VERIFIED.\n')
        f.write(f'Fast signatures: {len(peak)}; typical interval: {normal:.3f} µs; median envelope amplitude: {typical_amp:.3f} scope V; median envelope width: {typical_width:.3f} µs\n')
        f.write(f'Intentional low-going commands: {len(switch)}; median separation: {expected_switch:.3f} µs\n')
        f.write('Counts by type:\n'+result.type.value_counts().to_string()+'\n')
        f.write('\nNOTES: transition envelope widths are not direct logic low/high pulse widths. Large command dips can obscure fast transitions and generate apparent interval anomalies.\n')
    print((out/'summary.txt').read_text());print('Output:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('csv');p.add_argument('--out',default='encoder_full_results')
    p.add_argument('--bin-ns',type=float,default=100)
    p.add_argument('--prominence-scope-v',type=float,default=2)
    p.add_argument('--min-amplitude-scope-v',type=float,default=3)
    p.add_argument('--min-separation-us',type=float,default=10)
    p.add_argument('--interval-tolerance',type=float,default=.20)
    p.add_argument('--missing-multiple',type=float,default=1.7)
    p.add_argument('--amplitude-fraction',type=float,default=.60)
    p.add_argument('--width-fraction',type=float,default=.50)
    p.add_argument('--deep-fraction',type=float,default=.85)
    p.add_argument('--merge-gap-us',type=float,default=.6)
    p.add_argument('--min-switch-us',type=float,default=1.0)
    p.add_argument('--switch-tolerance',type=float,default=.12)
    p.add_argument('--switch-width-fraction',type=float,default=.65)
    p.add_argument('--low-max-relative-V',type=float,default=.5)
    analyze(p.parse_args())
