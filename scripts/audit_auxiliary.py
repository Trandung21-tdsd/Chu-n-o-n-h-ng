"""Inspect BAGLS provenance, segmentation pairing and cross-copy exact duplicates."""
import json, hashlib, re
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
root=Path('/Users/trandung/Downloads/TATO');report=Path('reports');summary={};meta_rows=[]
for name in ['BAGLS-VF_train','BAGLS-VF_test','archive-2']:
    p=root/name;files=[x for x in p.rglob('*') if x.is_file()]
    summary[name]={'files':len(files),'extensions':dict(Counter(x.suffix for x in files)), 'folders':dict(Counter(str(x.parent.relative_to(p)) for x in files))}
    if name=='archive-2':
        for f in files:
            if f.suffix=='.meta' and ('training' in f.parts or 'test' in f.parts):
                m=json.loads(f.read_text());meta_rows.append({'path':str(f),'original_split':'test' if 'test' in f.parts else 'train','frame':f.stem,**m})
print('metadata',len(meta_rows),flush=True)
m=pd.DataFrame(meta_rows);m.to_csv(report/'bagls_metadata.csv',index=False)
if len(m):
    summary['bagls_disorders']=m['Subject disorder status'].value_counts(dropna=False).to_dict()
    summary['bagls_video_id_overlap']=int((m.groupby('Video Id').original_split.nunique()>1).sum())
# Hash ALL pngs, distinguishing masks from inputs, without reading raw video.
files=sorted(f for name in summary if name in ['BAGLS-VF_train','BAGLS-VF_test','archive-2'] for f in (root/name).rglob('*.png'))
def inspect(f):
    rel=str(f.relative_to(root));role='mask' if ('mask' in rel.lower() or '_seg' in f.name.lower()) else 'image'
    return {'path':rel,'role':role,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(inspect,files))
df=pd.DataFrame(rows);dups=df[df.duplicated('sha256',keep=False)]
dups.to_csv(report/'bagls_exact_duplicates.csv',index=False)
summary['exact_hash_duplicate_extra_by_role']=df.groupby('role').apply(lambda x:int(x.duplicated('sha256').sum()),include_groups=False).to_dict()
summary['notes']=['BAGLS-VF derives from BAGLS: do not count copies as independent patients.', 'Identical all-black segmentation masks can be valid; identical mask alone does not imply duplicate input.', 'Video Id is a recording identifier; do not claim verified patient independence across recordings without subject mapping.', 'Disorder status in BAGLS is clinical metadata, not automatically a histopathological image diagnosis; kept out of CE-NBI classifier.']
(report/'auxiliary_audit.json').write_text(json.dumps(summary,indent=2,default=str));print(json.dumps(summary,indent=2,default=str))
