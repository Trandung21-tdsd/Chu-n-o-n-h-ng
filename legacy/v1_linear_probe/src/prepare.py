"""Audit original data without modifying it; split patients and duplicate components."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import imagehash
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path('/Users/trandung/Downloads/TATO/Larynx_CE-NBI_Dataset')
RARE = {'Amyloidosis','Bamboo_nodes','Namboo_node','Granuloma','Hemangioma','Hyperplasia','Inflammation','Nodule'}

def label_for(hist):
    return 'Other_benign' if hist in RARE else ('Papilloma' if hist == 'Papillomatosis' else hist)

class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while a != self.p[a]:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b: self.p[max(a,b)] = min(a,b)

def inspect_image(path):
    try:
        raw = path.read_bytes()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            w,h = im.size
            pixel = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            phash = str(imagehash.phash(im))
        return {'sha256':hashlib.sha256(raw).hexdigest(), 'pixel_hash':pixel, 'phash':phash,'width':w,'height':h,'error':''}
    except Exception as e: return {'error':str(e)}

def make_split(df, ratios=(.7,.15,.15), seed=42):
    if len(ratios)!=3 or not np.isclose(sum(ratios),1) or min(ratios)<=0:
        raise ValueError('Ratios must be three positive numbers summing to 1; 80:20:10 is invalid.')
    # Optimize ONLY class/group counts, never predictions or image features.
    patients = df[['patient_id','label','group']].drop_duplicates()
    if patients.groupby('patient_id').label.nunique().max()!=1: raise ValueError('Patient label conflict')
    classes=sorted(df.label.unique()); total=patients.label.value_counts().reindex(classes,fill_value=0)
    best=None
    outer=GroupShuffleSplit(n_splits=300,test_size=ratios[2],random_state=seed)
    for k,(rest,te) in enumerate(outer.split(patients,groups=patients.group)):
        rem=patients.iloc[rest]
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=ratios[1]/sum(ratios[:2]),random_state=seed+k).split(rem,groups=rem.group))
        parts=[rem.iloc[tr],rem.iloc[va],patients.iloc[te]]
        counts=[x.label.value_counts().reindex(classes,fill_value=0) for x in parts]
        if any((x==0).any() for x in counts): continue
        score=sum(float(np.mean((c/total-r)**2)) for c,r in zip(counts,ratios))
        if best is None or score<best[0]: best=(score,parts)
    if best is None: raise ValueError('Cannot produce three splits with all classes and isolated groups. Review rare classes.')
    mapping={p:s for s,part in zip(['train','val','test'],best[1]) for p in part.patient_id}
    out=df.copy();out['split']=out.patient_id.map(mapping)
    for col in ['patient_id','group','pixel_hash']:
        assert out.groupby(col).split.nunique().max()==1, f'Leakage: {col}'
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--out',type=Path,default=Path('data'));ap.add_argument('--ratios',nargs=3,type=float,default=[.7,.15,.15]);a=ap.parse_args()
    a.out.mkdir(exist_ok=True,parents=True);report=Path('reports');report.mkdir(exist_ok=True)
    if a.root == ROOT:
        subprocess.run([sys.executable,'scripts/check_patient_labels.py'],check=True)
    files=sorted(a.root.glob('*/*/Patient*/*.jpg'))
    rows=[]
    cachepath=a.out/'image_audit.csv'
    old=pd.read_csv(cachepath,keep_default_na=False).set_index('image_path').to_dict('index') if cachepath.exists() else {}
    def cached_inspect(f):
        entry=old.get(str(f))
        if entry and entry.get('sha256')==hashlib.sha256(f.read_bytes()).hexdigest(): return {k:entry[k] for k in ['sha256','pixel_hash','phash','width','height','error']}
        return inspect_image(f)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(f,meta) in enumerate(zip(files,pool.map(cached_inspect,files))):
            rel=f.relative_to(a.root);binary,hist,patient,_=rel.parts
            rows.append(dict(image_path=str(f),relative_path=str(rel),patient_id=patient,histology=hist,label=label_for(hist),binary=binary,**meta))
            if i%2000==0: print('audit',i,len(files),flush=True)
    df=pd.DataFrame(rows);df.to_csv(a.out/'image_audit.csv',index=False)
    df[df.error!=''].to_csv(report/'corrupt_images.csv',index=False)
    df=df[df.error==''].reset_index(drop=True)
    # Existing manifests: distinguish repeated class values from duplicate records.
    audit=[]; conflicts=[]
    for f in sorted((a.root/'manifests').glob('*.csv')):
        m=pd.read_csv(f);key='image_path'
        audit.append(dict(file=f.name,rows=len(m),duplicate_rows=int(m.duplicated().sum()),duplicate_paths=int(m.duplicated(key).sum()),patient_split_overlap=int((m.groupby('patient_id').split.nunique()>1).sum()) if 'split' in m else None))
        if 'label' in m:
            lookup=df.set_index('relative_path')['binary' if 'binary' in f.name else 'label']
            for row in m.itertuples():
                if row.image_path not in lookup or lookup[row.image_path]!=row.label:
                    conflicts.append({'file':f.name,'image_path':row.image_path,'given':row.label,'expected':lookup.get(row.image_path,'MISSING')})
            bad=m.groupby(key).label.nunique();bad=set(bad[bad>1].index)
            for keyval in bad: conflicts.append({'file':f.name,'image_path':keyval,'given':'CONFLICTING_ROWS','expected':''})
    pd.DataFrame(audit).to_csv(report/'label_file_audit.csv',index=False)
    pd.DataFrame(conflicts,columns=['file','image_path','given','expected']).to_csv(report/'label_conflicts.csv',index=False)
    if conflicts: raise ValueError('Manifest conflicts: review reports/label_conflicts.csv before training')
    badpatients=set(pd.read_csv(report/'patient_label_issues.csv').patient_id) if a.root==ROOT else set()
    df[df.patient_id.isin(badpatients)].to_csv(report/'quarantine_patient_labels.csv',index=False)
    df=df[~df.patient_id.isin(badpatients)].reset_index(drop=True)
    dup=df[df.duplicated('pixel_hash',keep=False)]
    dup.to_csv(report/'exact_duplicates.csv',index=False)
    badhash=df.groupby('pixel_hash').label.nunique();badhash=set(badhash[badhash>1].index)
    df[df.pixel_hash.isin(badhash)].to_csv(report/'quarantine.csv',index=False)
    df=df[~df.pixel_hash.isin(badhash)].reset_index(drop=True)
    uf=UnionFind(len(df))
    for col in ['patient_id','pixel_hash']:
        for ids in df.groupby(col).indices.values():
            for j in ids[1:]:uf.union(int(ids[0]),int(j))
    # All-pairs pHash radius <=4 via 5 pigeonhole bands; conservative grouping,
    # candidates are NOT automatically deleted (similar anatomy is not proof of duplicate).
    buckets={};near=[];hashes=[int(h,16) for h in df.phash]
    for i,h in enumerate(hashes):
        candidates=set()
        for band in range(5):
            key=(band,(h>>(band*13))&8191);candidates.update(buckets.get(key,[]))
        for j in candidates:
            distance=(h^hashes[j]).bit_count()
            if distance<=4 and df.iloc[i].pixel_hash!=df.iloc[j].pixel_hash:
                uf.union(i,j)
                near.append({'path_a':df.iloc[j].relative_path,'path_b':df.iloc[i].relative_path,'distance':distance,'cross_patient':df.iloc[j].patient_id!=df.iloc[i].patient_id})
        for band in range(5):buckets.setdefault((band,(h>>(band*13))&8191),[]).append(i)
    pd.DataFrame(near,columns=['path_a','path_b','distance','cross_patient']).to_csv(report/'near_duplicate_candidates.csv',index=False)
    df['group']=[uf.find(i) for i in range(len(df))]
    df=make_split(df,tuple(a.ratios))
    df['canonical']=~df.duplicated('pixel_hash')
    df.to_csv(a.out/'manifest.csv',index=False)
    dist=df.groupby(['split','label']).agg(images=('image_path','size'),patients=('patient_id','nunique'),unique_images=('pixel_hash','nunique'))
    dist.to_csv(report/'split_distribution.csv')
    summary=dict(images=len(df),patients=df.patient_id.nunique(),classes=sorted(df.label.unique()),exact_duplicate_extra=int(df.duplicated('pixel_hash').sum()),near_pairs=len(near),cross_patient_near_pairs=sum(x['cross_patient'] for x in near),ratios_requested=a.ratios,split_images=df.split.value_counts().to_dict(),split_patients=df.drop_duplicates('patient_id').split.value_counts().to_dict(),split_sha256=hashlib.sha256((a.out/'manifest.csv').read_bytes()).hexdigest())
    (report/'audit_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
