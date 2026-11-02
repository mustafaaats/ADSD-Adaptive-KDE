from pathlib import Path
import json,sys
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,cohen_kappa_score

root=Path(__file__).resolve().parent
key=json.loads((root/'answer_key_private.json').read_text());ids=[r['id'] for r in key]
y=np.array([r['true_label'] for r in key]);types=np.array([r['anomaly_type'] for r in key])

def read_response(path):
 text=Path(path).read_text().strip()
 if text.startswith('```'):
  lines=text.splitlines()
  if lines[-1].strip()!='```':raise ValueError('Unclosed JSON fence')
  text='\n'.join(lines[1:-1])
 if text.startswith('['):
  data=json.loads(text)
 else:
  data=[]
  for line in text.splitlines():
   line=line.strip()
   if not line:continue
   parts=[v.strip() for v in line.split('|',3)]
   if len(parts)!=4:raise ValueError('Expected id | score | label | reason on every line')
   rid,score,label,reason=parts
   data.append({'id':rid,'anomaly_score':float(score),'predicted_label':int(label),'reason':reason})
 if not isinstance(data,list) or len(data)!=len(ids):raise ValueError('Expected exactly 40 judgments')
 if len({r['id'] for r in data})!=len(ids) or {r['id'] for r in data}!=set(ids):raise ValueError('Missing, duplicate or unknown record IDs')
 byid={r['id']:r for r in data};data=[byid[i] for i in ids]
 for r in data:
  s=r['anomaly_score'];v=r['predicted_label']
  if not isinstance(s,(int,float)) or not np.isfinite(s) or not 0<=s<=100:raise ValueError('Invalid score')
  if v not in (0,1) or v!=int(s>=50):raise ValueError('Label must match score threshold 50')
  if not isinstance(r.get('reason'),str) or not r['reason'].strip():raise ValueError('Missing rationale')
 return data,np.array([r['anomaly_score'] for r in data]),np.array([r['predicted_label'] for r in data])

if len(sys.argv) not in [2,3]:raise SystemExit('Usage: python score_judge.py response_1.json [response_2.json]')
summary={};predictions=[]
for i,path in enumerate(sys.argv[1:],1):
 rows,s,p=read_response(path);predictions.append(p)
 summary[f'run_{i}']={'response_file':str(path),'n':len(y),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'ROC_AUC':roc_auc_score(y,s),'AP':average_precision_score(y,s),'confusion_matrix_rows_true_columns_predicted':confusion_matrix(y,p,labels=[0,1]).tolist(),'recall_by_type':{name:float(p[types==k].mean()) for k,name in [(1,'global'),(2,'variance_inflated_local'),(3,'dependency'),(4,'clustered')]},'normal_specificity':float((p[types==0]==0).mean())}
 joined=[{**k,**r} for k,r in zip(key,rows)]
 (root/f'scored_run_{i}.json').write_text(json.dumps(joined,indent=2))
if len(predictions)==2:
 p,q=predictions;v=cohen_kappa_score(p,q) if len(set(p.tolist()+q.tolist()))>1 else None
 summary['repeat_reliability']={'decision_agreement':float((p==q).mean()),'cohen_kappa':None if v is None or not np.isfinite(v) else float(v)}
(root/'evaluation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
