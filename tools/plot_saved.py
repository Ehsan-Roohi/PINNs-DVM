"""Plot saved shock-tube fields only; no training or reference regeneration."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['bgk','shakhov'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 d=np.genfromtxt(root/'data/shocktube'/f'{a.family}_matched_fields.csv',delimiter=',',names=True)
 fields=['rho','u','T','q','sigma','m','R','Delta','S']
 fig,axes=plt.subplots(3,3,figsize=(9,8))
 for ax,k in zip(axes.flat,fields):
  ax.plot(d['x'],d['parent_'+k],color='purple',ls='--',label='Initial PINN')
  ax.plot(d['x'],d['final_'+k],color='tab:blue',label='Refined PINN')
  take=np.linspace(0,len(d)-1,65,dtype=int)
  ax.plot(d['x'][take],d['DVM_'+k][take],color='tab:orange',marker='o',mfc='none',ms=3,ls='none',label='DVM reference')
  ax.set_xlabel('x');ax.set_ylabel(k);ax.grid(alpha=.2)
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
 fig.tight_layout(rect=(0,.05,1,1));a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out);plt.close(fig)
if __name__=='__main__':main()
