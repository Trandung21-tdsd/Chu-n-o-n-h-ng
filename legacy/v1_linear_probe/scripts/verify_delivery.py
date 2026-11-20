"""Integration checks against the actual audited dataset and selected checkpoint."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from throat_ai.predict import Predictor
m=pd.read_csv('data/manifest.csv')
for key in ['patient_id','group','pixel_hash']:assert m.groupby(key).split.nunique().max()==1
assert 'Patient092' not in set(m.patient_id)
q=pd.read_csv('reports/quarantine.csv');assert not set(q.pixel_hash)&set(m.pixel_hash)
near=pd.read_csv('reports/near_duplicate_candidates.csv');splits=m.set_index('relative_path').split
assert (near.path_a.map(splits)==near.path_b.map(splits)).all()
results=pd.read_csv('runs/benchmark_results.csv');assert len(results)==8;assert results.status.eq('completed').all()
assert results.n_val.nunique()==results.n_test.nunique()==1
best=json.load(open('runs/best_model.json'));assert results.loc[results.val_patient_macro_f1.idxmax(),'run_id']==best['run_id']
predictor=Predictor(best['checkpoint'])
pred=pd.read_csv('runs/'+best['run_id']+'/test_predictions.csv');r=pred.iloc[0]
p='/Users/trandung/Downloads/TATO/Larynx_CE-NBI_Dataset/'+r.relative_path
with Image.open(p) as im:
 result=predictor.predict(im,'CE-NBI');assert predictor.predict(im,'unknown')['status']=='out_of_scope'
for label,score in result['probabilities'].items():assert np.isclose(score,r['p_'+label],atol=1e-4),(label,score,r['p_'+label])
Path('reports/example_prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('PASS: patient/hash/near-duplicate isolation, quarantine, 8 runs, fixed evaluation sets, validation-only selection, checkpoint reload, prediction parity, out-of-scope gate.')
