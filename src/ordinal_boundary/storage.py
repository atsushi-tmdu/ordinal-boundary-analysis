from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import json
import os
import re
from .contracts import require, digest

def atomic_json(path:Path,value:object)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    except BaseException:
        temporary.unlink(missing_ok=True);raise

class CheckpointStore:
    def __init__(self,path:Path):self.path=Path(path).resolve()
    @contextmanager
    def transaction(self):
        self.path.mkdir(parents=True,exist_ok=True)
        lock=self.path/'.lock'
        try: fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError: raise RuntimeError('Checkpoint locked; check the prior process before manual recovery') from None
        os.close(fd)
        try:yield self
        finally:lock.unlink(missing_ok=True)
    def bind(self,identity:dict)->None:
        f=self.path/'identity.json'
        if f.exists():require(json.loads(f.read_text())==identity,"Checkpoint identity differs; use a new output directory")
        else:
            require(not any(p.name!='.lock' for p in self.path.iterdir()),"Unbound checkpoint directory is nonempty")
            atomic_json(f,identity)
    def _key(self,key:str)->Path:
        require(bool(re.fullmatch(r'[a-f0-9]{64}',key)),"Invalid checkpoint key")
        return self.path/(key+'.json')
    def read(self,key:str):
        p=self._key(key)
        if not p.exists():return None
        envelope=json.loads(p.read_text());require(digest(envelope['record'])==envelope['sha256'],"Checkpoint integrity failure")
        return envelope['record']
    def write(self,key:str,record:dict)->None:
        p=self._key(key);require(not p.exists(),"Checkpoint overwrite refused")
        atomic_json(p,{"sha256":digest(record),"record":record})

class AggregateExporter:
    FIELDS={'label','outcome','boundary','n','events','active_G','source_rows','sample_fingerprint','reference_k','step_k',
      'reference_llf','step_llf','reference_AIC','step_AIC','reference_BIC','step_BIC','delta_BIC','delta_AIC','LR_chi2','LR_df','LR_p',
      'step_log_OR','step_SE_cluster','step_OR_cluster','step_OR_low','step_OR_high','step_p_cluster'}
    @classmethod
    def write(cls,path:Path,rows:list[dict])->None:
        require(bool(rows),"Empty aggregate export")
        for r in rows:
            require(set(r)==cls.FIELDS,"Unexpected aggregate fields; potential row-level export refused")
            for k in ('label','outcome'):require(bool(re.fullmatch(r'[A-Za-z0-9_. -]{1,100}',r[k])),"Unsafe aggregate label")
            for k,v in r.items(): require(not isinstance(v,(list,dict,tuple)),"Nested or row-level output refused")
        import pandas as pd
        p=Path(path);raw=pd.DataFrame(rows).to_csv(index=False).encode()
        if p.exists():
            require(p.read_bytes()==raw,"Existing aggregate differs from validated rows");return
        p.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
