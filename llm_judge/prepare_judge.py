from pathlib import Path
import hashlib,json,argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--output',type=Path,required=True,help='Use a separate directory to preserve the archived evaluation.')
args=parser.parse_args()
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
root=args.repo.resolve()
d=np.load(root/'datasets/synthetic_constructed.npz');m=np.load(root/'datasets/synthetic_constructed_metadata.npz')
X,y=d['X'],d['y'];types=m['anomaly_type']
assert X.shape==(10000,10) and len(types)==len(y)
assert np.array_equal(types>0,y==1)
train,test=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=1)
# Reproduce the original seed-1, 5% labelled anomaly selection.
np.random.seed(1)
a,b=train_test_split(np.arange(len(y)),test_size=.3,stratify=y)
assert np.array_equal(a,train) and np.array_equal(b,test)
train_positive=np.flatnonzero(y[train]==1)
revealed=np.random.choice(train_positive,int(.05*len(train_positive)),replace=False)
reference=X[np.delete(train,revealed)]
median=np.median(reference,axis=0)
q1,q3=np.quantile(reference,[.25,.75],axis=0)
scale=(q3-q1)/1.349
assert (scale>0).all()
Z=(X-median)/scale
rng=np.random.default_rng(8240)
selected=np.concatenate([rng.choice(test[types[test]==k],20 if k==0 else 5,replace=False) for k in range(5)])
rng.shuffle(selected)
records=[{'id':f'R{i+1:03d}','features':np.round(Z[idx],4).tolist()} for i,idx in enumerate(selected)]
refZ=(reference-median)/scale
context={'feature_order':[f'x{i+1}' for i in range(10)],'reference_size':len(reference),'reference_quantiles':{str(q):np.round(np.quantile(refZ,q,axis=0),4).tolist() for q in [.01,.25,.5,.75,.99]},'reference_correlation_after_clipping_to_minus3_plus3':np.round(np.corrcoef(np.clip(refZ,-3,3),rowvar=False),3).tolist()}
instructions='''You are an auxiliary judge for a synthetic numerical anomaly-detection experiment. Evaluate only the numerical evidence below. Do not browse, use external files, or infer labels from record IDs or ordering. The reference pool is mostly normal but may contain anomalies. It is drawn only from training data, excluding the few revealed anomalies. Every candidate is held out from this reference pool.

Ten numerical features are expressed in robust standardised units: subtract the reference median and divide by reference IQR/1.349. The reference quantiles describe marginal values. The supplied correlation matrix describes reference relationships after clipping standardised values to [-3,3]. These are summaries, not the complete reference distribution. There are no semantic feature names or clinical meanings. Consider both extreme marginal values and unusual combinations of values. Judge each candidate separately; do not enforce a quota of normal or anomalous decisions. Do not assume every unusual-looking record must be anomalous. The candidate sampling frequency is not the population anomaly frequency.

For each record, return an anomaly score between 0 and 100 (a ranking score, not a calibrated probability), a binary label (1 if score >=50, otherwise 0), and a brief numerical rationale. Do not predict an anomaly subtype. Return a JSON array only, with exactly one object per candidate in the original order, using keys: id, anomaly_score, predicted_label, reason. Base all decisions on the supplied information. Keep each reason to at most 20 words.

REFERENCE_SUMMARY
'''
prompt=instructions+json.dumps(context,indent=2)+'\n\nCANDIDATES\n'+json.dumps(records,indent=2)+'\n'
(out/'judge_prompt.txt').write_text(prompt)
key=[{'id':r['id'],'source_index':int(idx),'true_label':int(y[idx]),'anomaly_type':int(types[idx])} for r,idx in zip(records,selected)]
(out/'answer_key_private.json').write_text(json.dumps(key,indent=2))
protocol={'split_seed':1,'sampling_seed':8240,'candidate_count':40,'sampling':{'normal':20,'global':5,'variance_inflated_local':5,'dependency':5,'clustered':5},'reference_supervision_ratio':.05,'reference_pool_count':len(reference),'threshold':50,'repeats':2,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'dataset_sha256':hashlib.sha256((root/'datasets/synthetic_constructed.npz').read_bytes()).hexdigest(),'primary_metrics':['balanced_accuracy','precision','recall','f1'],'secondary_metrics':['ROC_AUC','AP'],'reliability_metrics':['decision_agreement','Cohen_kappa'],'limitations':['balanced enriched diagnostic sample; not representative prevalence','one split and five anomalies per subtype','LLM numeric judgments are auxiliary and not expert validation','no inference that LLM validates ADSD bandwidth superiority','no prompt revision after viewing labels or outcomes'],'judge_metadata_required':['displayed_model_name','evaluation_date','tools_or_reasoning_mode','fresh_chat_per_repeat']}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
assert not set(train)&set(selected) and len(set(selected))==40
print('Prepared 40 blinded test records; training reference:',len(reference),'records.')
print('For replication, use judge_prompt_plain.txt only; do not reveal the answer key.')

intro=instructions[:instructions.index('For each record, return')]
intro+='For every candidate, give an anomaly score from 0 to 100 (a ranking score, not a calibrated probability), a predicted label (1 if score >=50, otherwise 0), and a brief reason of at most 12 words. Do not predict anomaly subtypes. Return exactly 40 plain-text lines, one per record in the input order. Separate the four fields with a vertical bar. Do not use JSON, square brackets, Markdown tables, code fences, or introductory text. Format each line as: record_id | anomaly_score | predicted_label | reason. Use the actual record IDs supplied below.\n\n'
lines=[intro,'FEATURE ORDER: '+','.join(context['feature_order']),'REFERENCE POOL SIZE: '+str(context['reference_size']),'REFERENCE QUANTILES (one value for each feature):']
for q,v in context['reference_quantiles'].items():lines.append(q+': '+','.join(map(str,v)))
lines.append('REFERENCE CORRELATION MATRIX (standardised values clipped to -3,+3):')
for f,row in zip(context['feature_order'],context['reference_correlation_after_clipping_to_minus3_plus3']):lines.append(f+': '+','.join(map(str,row)))
lines.extend(['','CANDIDATES:','id,'+','.join(context['feature_order'])])
for r in records:lines.append(r['id']+','+','.join(map(str,r['features'])))
plain='\n'.join(lines)+'\n'
(out/'judge_prompt_plain.txt').write_text(plain)
