"""Supervised experiment queue: clean, fixed, curriculum; fail on any failed stage."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .generate_all import SUITES
from .audit_data import audit

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/pi05_initial_v2_20261113'
DATA=ROOT/'data/initial_joint_v2_20261113'
STEPS=30000

def main():
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'.queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(stage,**extra):
        value={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),**extra}
        p=RUN/'status.tmp';p.write_text(json.dumps(value,indent=2));p.replace(RUN/'status.json');print(json.dumps(value),flush=True)
    def execute(cmd,log):
        state('running',command=cmd,log=str(log))
        with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    clean=[str(Path('/home/tushaar/data/LIBERO-datasets')/s) for s in SUITES]
    try:
        for mode in ['clean','fixed','curriculum']:
            roots=list(clean)
            if mode!='clean':
                if not (DATA/'COMPLETE.json').exists():
                    state('waiting_for_dataset',dataset=str(DATA))
                    while not (DATA/'COMPLETE.json').exists():
                        # Stop if the generation session failed rather than silently waiting forever.
                        active=subprocess.run(['pgrep','-u',str(os.getuid()),'-f','^robo_env/bin/python -u -m pi05_pipeline.generate_all'],stdout=subprocess.DEVNULL).returncode==0
                        if not active:raise RuntimeError('Dataset incomplete and generator is not running; inspect generation log')
                        time.sleep(30)
                audit(DATA,RUN/'dataset_audit.json');roots += [str(DATA/s) for s in SUITES]
            out=ROOT/'checkpoints'/('pi05_initial_v2_'+mode+'_seed0')
            final=out/f'step_{STEPS:06d}'
            if not final.exists():
                cmd=[sys.executable,'-u','-m','pi05_pipeline.train','--data',*roots,'--out',str(out),'--mode',mode,
                     '--steps',str(STEPS),'--seed','0','--batch-size','4','--accumulation','4','--save-every','500','--validation-batches','32']
                if out.exists():
                    latest=out/'latest.txt'
                    if not latest.exists():raise RuntimeError(f'Incomplete run without checkpoint: {out}')
                    cmd+=['--resume',str(out/latest.read_text().strip())]
                execute(cmd,RUN/(mode+'_train.log'))
            if not (final/'adapter.safetensors').is_file():raise RuntimeError('Final adapter missing')
            # Fixed final step is selected before seeing benchmark outcomes.
            for benchmark in ['standard','plus']:
                for suite in SUITES:
                    dest=RUN/(mode+'_'+benchmark+'_'+suite)
                    if dest.exists():
                        if (dest/'results.json').exists() and json.loads((dest/'results.json').read_text()).get('complete'):continue
                        raise RuntimeError(f'Incomplete evaluation preserved for diagnosis: {dest}')
                    execute([sys.executable,'-u','-m','pi05_pipeline.evaluate','--checkpoint',str(final),'--benchmark',benchmark,
                        '--suite',suite,'--out',str(dest)],RUN/(mode+'_evaluate.log'))
        state('complete')
    except BaseException as e:
        state('failed',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
