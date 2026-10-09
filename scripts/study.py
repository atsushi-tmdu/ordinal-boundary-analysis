from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from ordinal_boundary.contracts import require, ContractError
from ordinal_boundary.studies.primary import StudyContract, StudyRunner, TASKS, OUTCOMES
from ordinal_boundary.studies.simulated import simulated_eligible
from ordinal_boundary.storage import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--era',choices=('2008-2014','2019-2023'),default='2008-2014')
    parser.add_argument('--task',choices=TASKS,default='q1')
    parser.add_argument('--outcome',choices=OUTCOMES,default='intubation')
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--private-output',type=Path)
    args=parser.parse_args(argv)
    config=ROOT/'configs'/('sjtrem_'+args.era.replace('-','_')+'.json')
    contract=StudyContract.from_file(config)
    if not args.synthetic and not args.execute:
        print(json.dumps({'status':'INSPECTION_ONLY','task':args.task,'outcome':args.outcome,'spec':contract.for_task(args.task,args.outcome).spec()},default=str))
        return 0
    try:
        require(not(args.synthetic and (args.execute or args.input)),'Incompatible synthetic/private modes')
        if args.synthetic:
            frame=simulated_eligible(args.era)
            result=StudyRunner(contract).execute(frame,args.task,args.outcome)
            print(json.dumps({'status':'SYNTHETIC_COMPLETE','task':args.task,'outcome':args.outcome,'n_comparisons':result.get('n_comparisons',1)}))
            return 0
        require(args.input is not None and args.private_output is not None,'Private paths required')
        source=args.input.resolve();destination=args.private_output.resolve()
        require(ROOT not in (source,destination) and ROOT not in source.parents and ROOT not in destination.parents,'Private files cannot be stored inside the repository')
        require(source!=destination and source not in destination.parents and destination not in source.parents,'Input/output overlap')
        require(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),'Private output directory must be empty')
        frame=pd.read_csv(source,dtype={'sid':'string','hospital':'string'})
        result=StudyRunner(contract).execute(frame,args.task,args.outcome)
        atomic_json(destination/(args.era+'_'+args.outcome+'_'+args.task+'.json'),result)
        print(json.dumps({'status':'PRIVATE_OUTPUT_WRITTEN','task':args.task}))
        return 0
    except (ContractError,ValueError,KeyError,OSError,RuntimeError):
        print('STOP: private input or model contract failed',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
