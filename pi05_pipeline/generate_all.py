"""Sequential generation with immutable version manifests and restart checks."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import h5py

SUITES=['libero_spatial','libero_object','libero_goal','libero_10']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/tushaar/data/LIBERO-datasets'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--demos',type=int,default=0)
    p.add_argument('--seed',type=int,default=1729);a=p.parse_args()
    plan=[]
    for suite in SUITES:
        files=sorted((a.source/suite).glob('*.hdf5'))
        if len(files)!=10:raise ValueError(f'Expected ten task files in {suite}')
        for path in files:plan.append({'source':str(path.resolve()),'suite':suite,'bytes':path.stat().st_size})
    manifest={'plan':plan,'demos':a.demos,'seed':a.seed,'levels':5,'repeats':1,
        'generator_sha256':hashlib.sha256(Path(__file__).with_name('initial_data.py').read_bytes()).hexdigest()}
    a.output.mkdir(parents=True,exist_ok=True);mp=a.output/'plan.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Generation plan changed; use a new version')
    mp.write_text(json.dumps(manifest,indent=2));records=[]
    for job in plan:
        folder=a.output/job['suite'];folder.mkdir(exist_ok=True)
        source=Path(job['source']);report_path=folder/(source.stem.removesuffix('_demo')+'_generation.json')
        if not report_path.exists():
            with (folder/(source.stem+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u','-m','pi05_pipeline.initial_data','--source',str(source),
                    '--output',str(folder),'--demos',str(a.demos),'--seed',str(a.seed)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Generation failed: {source}; inspect its log')
        report=json.loads(report_path.read_text())
        if not report.get('complete') or report['generator_sha256']!=manifest['generator_sha256']:
            raise ValueError(f'Incomplete or mismatched report: {report_path}; preserve and diagnose it before restart')
        if report['accepted']==0:raise ValueError(f'No accepted recoveries: {source}')
        with h5py.File(folder/source.name,'r') as f:
            if not f['data'].attrs.get('generation_complete'):raise ValueError('Incomplete HDF5')
        records.append({**job,'accepted':report['accepted'],'complete':True})
        tmp=a.output/'status.tmp';tmp.write_text(json.dumps(records,indent=2));tmp.replace(a.output/'status.json')
        print(json.dumps({'completed_tasks':len(records),'total_tasks':len(plan),'job':records[-1]}),flush=True)
    (a.output/'COMPLETE.json').write_text(json.dumps({'tasks':len(records),'accepted':sum(r['accepted'] for r in records),'manifest':manifest},indent=2))

if __name__=='__main__':main()
