"""Read-only generation, training and evaluation status; no ML imports."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    run=ROOT/'results/pi05_initial_v2_20261113';data=ROOT/'data/initial_joint_v2_20261113'
    state=run/'status.json'
    if state.exists():print('Queue:',json.loads(state.read_text()))
    checked=accepted=attempts=finished=0
    for p in data.glob('*/*_generation.json'):
        r=json.loads(p.read_text());checked+=sum('clean_replay_success' in e for e in r['events'])
        accepted+=r['accepted'];attempts+=sum('level' in e for e in r['events']);finished+=bool(r.get('complete'))
    print(f'Dataset: {finished}/40 tasks finished; {checked}/2000 source demos checked; {accepted}/{attempts} attempted recoveries accepted')
    print(f"{'Model':12s} {'Optimizer steps':20s} {'Recent sec/step':18s} {'Standard success':20s} {'Robot-init success'}")
    for mode in ['clean','fixed','curriculum']:
        folder=ROOT/'checkpoints'/('pi05_initial_v2_'+mode+'_seed0');metrics=folder/'metrics.jsonl'
        rows=[json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
        step=rows[-1]['step'] if rows else 0;recent=rows[-50:]
        secs=f"{sum(r['seconds'] for r in recent)/len(recent):.2f}" if recent else '-'
        results=[]
        for benchmark,total in [('standard',2000),('plus',1550)]:
            n=s=complete=0
            for p in run.glob(f'{mode}_{benchmark}_*/results.json'):
                value=json.loads(p.read_text());n+=value['completed_episodes'];s+=value.get('successes',sum(r['success'] for r in value['results']));complete+=bool(value.get('complete'))
            results.append(f'{s/n:.2%} ({n}/{total})' if n else 'not started')
        print(f'{mode:12s} {str(step)+"/30000":20s} {secs:18s} {results[0]:20s} {results[1]}')

if __name__=='__main__':main()
