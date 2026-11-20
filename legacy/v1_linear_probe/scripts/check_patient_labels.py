from pathlib import Path
import pandas as pd
root=Path('/Users/trandung/Downloads/TATO/Larynx_CE-NBI_Dataset')
df=pd.read_excel(root/'Patients_List_Updated_Final.xlsx');df=df[df.Patient.astype(str).str.match(r'^Patient\d+$')].copy()
normalize=lambda s: str(s).strip().replace('’',"'").lower().replace(' ','_').replace("'",'')
alias={'squamous_hyperplasia':'hyperplasia','papilloma':'papillomatosis','bamboo_node':'namboo_node','bamboo_nodes':'namboo_node'}
issues=[]
for pid,group in df.groupby('Patient'):
    if group[['Histopathology','Type of Lesion','Leukoplakia']].drop_duplicates().shape[0]>1:issues.append((pid,'conflicting patient rows'))
    row=group.iloc[0];files=list(root.glob(f'*/*/{pid}/*.jpg'))
    if len(files)!=int(row['Number of Images']):issues.append((pid,'image count mismatch'))
    for f in files:
        actual=normalize(f.parts[-3]);expected=normalize(row.Histopathology);expected=alias.get(expected,expected)
        if actual!=expected or f.parts[-4]!=row['Type of Lesion']:issues.append((pid,f'label mismatch: {actual} vs {expected}'));break
pd.DataFrame(issues,columns=['patient_id','issue']).to_csv('reports/patient_label_issues.csv',index=False)
summary={'patients':df.Patient.nunique(),'rows':len(df),'duplicate_patient_rows':int(df.duplicated('Patient').sum()),'issues':len(issues)}
import json
Path('reports/patient_label_audit.json').write_text(json.dumps(summary,indent=2));print(summary);print(issues[:20])
# Conflicting patients are quarantined by prepare, never relabeled automatically.
