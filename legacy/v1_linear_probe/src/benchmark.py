"""Frozen ImageNet encoder + trained linear head. Test never selects a model."""
import argparse, hashlib, json, time, uuid, random, sys, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from .model import backbone, preprocess, device, PREPROCESS_VERSION

def metrics(y,p):
    k=p.shape[1];pred=p.argmax(1);labels=list(range(k))
    result=dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=labels,average='macro',zero_division=0),balanced_accuracy=recall_score(y,pred,labels=labels,average='macro',zero_division=0),macro_precision=precision_score(y,pred,labels=labels,average='macro',zero_division=0))
    try:result['macro_auroc']=roc_auc_score(y,p,multi_class='ovr',labels=labels)
    except ValueError:result['macro_auroc']=None
    return result

def patient_metrics(frame,p):
    tmp=pd.DataFrame(p);tmp['patient_id']=frame.patient_id.to_numpy();tmp['y']=frame.y.to_numpy()
    if tmp.groupby('patient_id').y.nunique().max()!=1:raise ValueError('Patient labels conflict')
    grouped=tmp.groupby('patient_id').mean(numeric_only=True)
    return metrics(grouped.y.to_numpy().astype(int),grouped.drop(columns='y').to_numpy()),grouped

def features(df,name,clahe,cache,dev,batch=64):
    signature=hashlib.sha256((df[['image_path','pixel_hash']].to_csv(index=False)+name+str(clahe)+PREPROCESS_VERSION).encode()).hexdigest()[:20]
    path=cache/f'{name}_{signature}.npy';weights_path=cache/f'{name}_encoder.pt'
    m,dim=backbone(name,pretrained=not weights_path.exists())
    if weights_path.exists():m.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
    else:torch.save(m.state_dict(),weights_path)
    if path.exists():return np.load(path),weights_path,0.
    m.to(dev).eval();parts=[];start=time.time()
    def read(path):
        with Image.open(path) as im:return preprocess(im,clahe)
    with torch.inference_mode(),ThreadPoolExecutor(max_workers=6) as pool:
        for i in range(0,len(df),batch):
            x=torch.stack(list(pool.map(read,df.image_path.iloc[i:i+batch]))).to(dev)
            parts.append(m(x).cpu().numpy())
            if i%(batch*20)==0:print(name,'CLAHE',clahe,'features',i,'/',len(df),flush=True)
    result=np.concatenate(parts);np.save(path,result);del m
    if dev.type=='mps':torch.mps.empty_cache()
    return result,weights_path,time.time()-start

def train_head(x,df,classes,seed,epochs,patience,lr,run):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(4)
    tr=df.index[df.split=='train'].to_numpy();va=df.index[df.split=='val'].to_numpy()
    mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-5)
    z=torch.from_numpy((x-mean)/std);y=torch.tensor(df.y.to_numpy(),dtype=torch.long)
    head=nn.Linear(x.shape[1],len(classes));optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
    counts=np.bincount(df.loc[tr,'y'],minlength=len(classes));weights=torch.tensor(len(tr)/(len(classes)*counts),dtype=torch.float)
    lossfn=nn.CrossEntropyLoss(weight=weights);best=-1.;stale=0;history=[];bestepoch=0
    for epoch in range(1,epochs+1):
        head.train();order=tr[np.random.permutation(len(tr))];loss_sum=0.
        for ids in np.array_split(order,max(1,int(np.ceil(len(order)/256)))):
            optimizer.zero_grad();loss=lossfn(head(z[ids]),y[ids]);loss.backward();optimizer.step();loss_sum+=loss.item()*len(ids)
        head.eval()
        with torch.inference_mode():pv=head(z[va]).softmax(1).numpy()
        met,_=patient_metrics(df.loc[va],pv);score=met['macro_f1']
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(tr),val_patient_macro_f1=score,**{'val_image_'+k:v for k,v in metrics(y[va].numpy(),pv).items()}))
        if score>best+1e-4:
            best=score;stale=0;bestepoch=epoch;torch.save(head.state_dict(),run/'head_best.pt')
        else:stale+=1
        if stale>=patience:break
    pd.DataFrame(history).to_csv(run/'epochs.csv',index=False)
    head.load_state_dict(torch.load(run/'head_best.pt',weights_only=True));head.eval()
    with torch.inference_mode():p=head(z).softmax(1).numpy()
    return head,mean,std,p,bestepoch,epoch,stale>=patience

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=Path('data/manifest.csv'));ap.add_argument('--out',type=Path,default=Path('runs'));ap.add_argument('--models',nargs='+',default=['resnet18','mobilenet_v3_small']);ap.add_argument('--seeds',nargs='+',type=int,default=[42]);ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=10);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--batch',type=int,default=64);a=ap.parse_args()
    a.out.mkdir(exist_ok=True,parents=True);cache=Path('data/features');cache.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(a.manifest).reset_index(drop=True);classes=sorted(df.label.unique());df['y']=df.label.map({v:i for i,v in enumerate(classes)})
    for col in ['patient_id','group','pixel_hash']:
        if df.groupby(col).split.nunique().max()!=1:raise ValueError(f'Leakage: {col}')
    fingerprint=hashlib.sha256(a.manifest.read_bytes()).hexdigest();dev=device();print('device',dev,flush=True)
    records=[];pending=[];suite=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    # Fit all configurations and choose winner BEFORE test evaluation.
    for name in a.models:
        for clahe in [False,True]:
            x,weights_path,feature_seconds=features(df,name,clahe,cache,dev,a.batch)
            for dedup in [False,True]:
                keep=((df.split=='train') if not dedup else pd.Series(False,index=df.index))|df.canonical
                sub=df.loc[keep].copy().reset_index(drop=True);sx=x[keep.to_numpy()]
                for seed in a.seeds:
                    rid=f'{suite}_{name}_clahe{int(clahe)}_dedup{int(dedup)}_s{seed}';run=a.out/rid;run.mkdir()
                    config=dict(run_id=rid,model=name,clahe=clahe,deduplicate_train=dedup,seed=seed,mode='frozen_imagenet_encoder_linear_probe',preprocess=PREPROCESS_VERSION,split_sha256=fingerprint,epochs_max=a.epochs,patience=a.patience,lr=a.lr,weight_decay=.01,device=str(dev),torch_version=str(torch.__version__),python=sys.version,platform=platform.platform(),feature_seconds=feature_seconds)
                    (run/'config.json').write_text(json.dumps(config,indent=2));start=time.time()
                    try:
                        head,mean,std,p,bestepoch,epochs,stopped=train_head(sx,sub,classes,seed,a.epochs,a.patience,a.lr,run)
                        val=sub.split=='val';vm,_=patient_metrics(sub[val],p[val]);im=metrics(sub.loc[val,'y'],p[val])
                        ckpt=dict(config=config,classes=classes,head=head.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),encoder=torch.load(weights_path,weights_only=True,map_location='cpu'))
                        torch.save(ckpt,run/'model.pt')
                        record=dict(**config,status='completed',best_epoch=bestepoch,epochs_run=epochs,early_stopped=stopped,train_seconds=time.time()-start,n_train=int((sub.split=='train').sum()),n_val=int(val.sum()),n_test=int((sub.split=='test').sum()),**{'val_patient_'+k:v for k,v in vm.items()},**{'val_image_'+k:v for k,v in im.items()})
                        pending.append((run,sub,p,record));records.append(record)
                        print(rid,'val patient macro F1',round(vm['macro_f1'],4),'epochs',epochs,flush=True)
                    except Exception as exc:
                        records.append(dict(**config,status='failed',error=repr(exc)));pd.DataFrame(records).to_csv(a.out/f'{suite}_benchmark.csv',index=False);raise
                    pd.DataFrame(records).to_csv(a.out/f'{suite}_benchmark.csv',index=False)
    # Validation only, one predeclared ranking criterion. Test is now unlocked once.
    winner=max(pending,key=lambda item:item[3]['val_patient_macro_f1'])
    selection={'run_id':winner[3]['run_id'],'checkpoint':str((winner[0]/'model.pt').resolve()),'criterion':'val_patient_macro_f1','split_sha256':fingerprint,'test_used_for_selection':False}
    (a.out/'best_model.json').write_text(json.dumps(selection,indent=2))
    for run,sub,p,record in pending:
        for split in ['val','test']:
            mask=sub.split==split;frame=sub[mask];probs=p[mask];met=metrics(frame.y,probs);pm,grouped=patient_metrics(frame,probs)
            if split=='test':record.update({'test_image_'+k:v for k,v in met.items()});record.update({'test_patient_'+k:v for k,v in pm.items()})
            pred=frame[['relative_path','patient_id','label']].copy();pred['prediction']=[classes[i] for i in probs.argmax(1)]
            for i,c in enumerate(classes):pred['p_'+c]=probs[:,i]
            pred.to_csv(run/f'{split}_predictions.csv',index=False)
            pd.DataFrame(confusion_matrix(frame.y,probs.argmax(1),labels=range(len(classes))),index=classes,columns=classes).to_csv(run/f'{split}_confusion.csv')
            pd.DataFrame(classification_report(frame.y,probs.argmax(1),labels=range(len(classes)),target_names=classes,output_dict=True,zero_division=0)).T.to_csv(run/f'{split}_per_class.csv')
            grouped.to_csv(run/f'{split}_patient_probabilities.csv')
        record['selected_by_validation']=record['run_id']==selection['run_id']
    result=pd.DataFrame(records);result.to_csv(a.out/f'{suite}_benchmark.csv',index=False)
    master=a.out/'benchmark_results.csv'
    if master.exists():result=pd.concat([pd.read_csv(master),result],ignore_index=True)
    result.to_csv(master,index=False)
    print('SELECTED',json.dumps(selection),flush=True)

if __name__=='__main__':main()
