import numpy as np
import pandas as pd
import pytest
from PIL import Image
from throat_ai.prepare import make_split,UnionFind,label_for
from throat_ai.model import preprocess

def sample():
    rows=[]
    for c in ['A','B']:
        for i in range(20):
            for j in range(2):rows.append(dict(patient_id=f'{c}{i}',group=f'{c}{i}',label=c,pixel_hash=f'{c}{i}-{j}'))
    return pd.DataFrame(rows)

def test_patient_and_duplicate_isolation():
    d=sample();d.loc[d.patient_id=='A1','group']='A0'
    out=make_split(d)
    for col in ['patient_id','group','pixel_hash']:assert out.groupby(col).split.nunique().max()==1
    assert all(set(x.label)=={'A','B'} for _,x in out.groupby('split'))
    pd.testing.assert_frame_equal(out,make_split(d))

def test_invalid_ratio_rejected():
    with pytest.raises(ValueError):make_split(sample(),(.8,.2,.1))

def test_patient_conflict_rejected():
    d=sample();d.loc[0,'label']='B'
    with pytest.raises(ValueError):make_split(d)

def test_union_transitivity():
    u=UnionFind(4);u.union(0,1);u.union(2,3);u.union(1,3);assert len({u.find(i) for i in range(4)})==1

def test_clahe_and_input_mode():
    arr=np.tile(np.arange(80,160,dtype=np.uint8),(64,1));im=Image.fromarray(arr)
    a=preprocess(im);b=preprocess(im,True)
    assert a.shape==b.shape==(3,224,224)
    assert np.isfinite(b.numpy()).all();assert not np.allclose(a,b)
    assert label_for('Namboo_node')=='Other_benign'
