"""Plot verified saved M2 stage endpoints and histories; no solver/checkpoint imports."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release-data',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
 ev=a.release_data/'evaluation';e=json.loads((ev/'RESULT.json').read_text());inputs={str(ev/'RESULT.json'):sha(ev/'RESULT.json')}
 seeds=[141001,141002,141003]; stages=['adam20000','ssb500','fixed500']; dirs=['adam','ssb','fixed']; colors=['#0072B2','#D55E00','#009E73'];styles=['-','--','-.'];curves={};hist={};metrics={}
 for si,s in enumerate(seeds):
  for j,(st,d) in enumerate(zip(stages,dirs)):
   p=ev/f'seed{s}_{st}_core.npz';assert sha(p)==e['output_hashes'][p.name];inputs[str(p)]=sha(p)
   with np.load(p,allow_pickle=False) as z:x=z['x'].copy();y=z['fields'].copy();r=z['reference'].copy()
   assert len(x)==81 and np.all(np.diff(x)>0) and np.all(np.isfinite(y)) and x[0]>=-10 and x[-1]<=10
   curves[s,st]=(x,y,r)
   rec=next(v for v in e['stage_records'] if v['seed']==s and v['stage']==st);assert rec['completed']
   metrics[f'{s}_{st}']={}
   for k,idx in [('q',3),('sigma',4)]:
    v=float(100*np.sqrt(np.trapezoid((y[:,idx]-r[:,idx])**2,x)/np.trapezoid(r[:,idx]**2,x)))
    expected=rec['windows']['core']['errors'][k]['relative_L2_percent'];assert np.isclose(v,expected,rtol=1e-10,atol=1e-12)
    metrics[f'{s}_{st}'][k]=v
   p=a.release_data/f'seed_{s}'/d/'history.jsonl';inputs[str(p)]=sha(p);rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];rows=[v for v in rows if d=='adam' or v.get('accepted') is True]
   steps=np.array([v['step'] for v in rows]);loss=np.array([v['loss'] for v in rows]);assert len(steps)>0 and np.all(np.diff(steps)>0) and np.all(loss>0)
   assert int(steps[-1])==([20000,500,500][j])
   if j:assert len(steps)==500
   hist[s,st]=(steps,loss)
 base_x,_,base_r=curves[seeds[0],stages[0]]
 for x,y,r in curves.values():assert np.array_equal(x,base_x) and np.array_equal(r,base_r)
 plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,ax=plt.subplots(3,3,figsize=(7.1,7.1));letters='abcdefghi'
 for j,st in enumerate(stages):
  for si,s in enumerate(seeds):
   x,y,r=curves[s,st];n,l=hist[s,st];ax[0,j].semilogy(n,l,color=colors[si],ls=styles[si],lw=.7 if j==0 else 1.25,alpha=.75 if j==0 else 1)
   for row,idx in [(1,3),(2,4)]:ax[row,j].plot(x,y[:,idx],color=colors[si],ls=styles[si],lw=1.25)
  for row,idx in [(1,3),(2,4)]:
   inds=np.arange(0,len(x),3);ax[row,j].plot(x[inds],r[inds,idx],linestyle='none',marker='o',ms=3.3,mfc='none',mec='black',mew=.7);ax[row,j].set(xlabel='$x$',xlim=(-10,10))
  ax[0,j].set(xlabel=['Adam update','SSB update','Split-half SSB update'][j],ylabel=['$J_{\\mathrm{batch}}$','$J_{\\mathrm{SSB}}$','$J_{\\mathrm{split}}$'][j])
 for row,idx,label in [(1,3,'$q_x$'),(2,4,'$\\sigma_{xx}$')]:
  vals=np.concatenate([v[1][:,idx] for v in curves.values()]+[v[2][:,idx] for v in curves.values()]);lo,hi=vals.min(),vals.max();pad=.06*(hi-lo)
  for j in range(3):ax[row,j].set(ylim=(lo-pad,hi+pad),ylabel=label)
 for k,z in enumerate(ax.flat):
  z.grid(alpha=.17,lw=.5);z.tick_params(direction='in');z.text(0,1.04,f'({letters[k]})',transform=z.transAxes,ha='left',va='bottom',fontsize=10)
 handles=[Line2D([],[],color=colors[i],ls=styles[i],lw=1.3,label=f'Seed {s}') for i,s in enumerate(seeds)]+[Line2D([],[],ls='none',marker='o',mfc='none',mec='black',ms=4,label='Shakhov DVM')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
 fig.subplots_adjust(left=.095,right=.985,top=.965,bottom=.18,hspace=.55,wspace=.55)
 a.outdir.mkdir(parents=True,exist_ok=True);outputs={}
 for ext in ['pdf','svg','png']:
  p=a.outdir/f'M2_OPTIMIZER_EFFECT.{ext}';fig.savefig(p,dpi=600,bbox_inches='tight',pad_inches=.04);outputs[str(p)]=sha(p)
 plt.close(fig)
 receipt={'scope':'Saved-only plotting; no inference/training; all three seeds included.','inputs':inputs,'outputs':outputs,'generator_sha256':sha(Path(__file__)),'metrics_verified_against_saved_result':metrics,'reference_and_x_identical_all9':True,'history_endpoints':{f'{s}_{st}':{'count':len(v[0]),'first_step':int(v[0][0]),'last_step':int(v[0][-1]),'first_loss':float(v[1][0]),'last_loss':float(v[1][-1])} for (s,st),v in hist.items()},'reference':'Existing saved linear interpolation onto unchanged model core nodes; no new phase or amplitude adjustment.','caption':'Mach-2 Shakhov optimisation and transport moments for all three independent initialisations. Columns show Adam, SSB and split-half SSB stages. Top row: recorded stage objectives; middle and bottom rows: signed heat flux and stress at each stage endpoint. DVM symbols show the saved reference interpolated onto common model nodes in [-10,10]. Physical ordinate limits are identical across columns. Objectives and quadrature change between stages; no connected loss curve or optimizer-only causal attribution is implied. Mean q/stress errors (%) change from20.1871/16.5819 to2.11274/1.08211 and1.03734/1.27447.'}
 (a.outdir/'M2_OPTIMIZER_EFFECT_QA.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'outputs':outputs,'metric_pairs_verified':len(metrics)}))
if __name__=='__main__':main()