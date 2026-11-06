"""CPU-only tests; temporary files stay in the explicitly supplied test directory."""
import copy, json, os, signal, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import continue_characteristic as c

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parent/'CPU_TEST';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.p=Path(self.tmp.name)
        cfg=dict(collision='bgk',model=dict(hidden=2,depth=1,time_nodes=2,grid=[4,4,4],vmax=7.,kn=1.,tf=.02),
                 lr=.001,adam_steps=1,ssb_steps=0,batch=2,fixed_points=2,chunk=1,threads=1,train_seconds=90,seed=71)
        c.native.run(cfg,self.p/'parent','cpu');self.parent=self.p/'parent/STATE.pt'
        self.base=torch.load(self.parent,weights_only=False)
        self.plan=dict(parent_sha256=c.native.sha(self.parent),sources=self.base['sources'],additional_updates=2,train_seconds=90)
    def tearDown(self):self.tmp.cleanup()
    def read(self,n):return torch.load(self.p/n/'STATE.pt',weights_only=False)
    def test_real_split_equivalence(self):
        a=c.run(self.plan,self.parent,self.p/'full');c.run(self.plan,self.parent,self.p/'split',stop_after=1)
        if a['status']!='COMPLETE':self.fail('Tiny real SSB fixture must complete: '+str(a))
        b=c.run(self.plan,self.parent,self.p/'split',resume=True)
        self.assertEqual(b['status'],'COMPLETE');x=self.read('full');y=self.read('split')
        for key in ('model','adam','ssb','rng','frozen','history','config','sources','adam_steps','ssb_steps'):
            self.assertTrue(equal(x[key],y[key]),key)
        for key in ('adam','frozen','config','sources','adam_steps','rng'):
            self.assertTrue(equal(x[key],self.base[key]),key)
    def test_reject_and_interrupt_transaction(self):
        for kind in ('reject','interrupt'):
            def trial(opt,x,fg):
                fg(x+.01);opt.H.add_(.2);torch.rand(1)
                if kind=='interrupt':signal.raise_signal(signal.SIGTERM)
                return dict(accepted=False,reason='test_rejection')
            with patch.object(c.native.DenseSSB,'step',trial):r=c.run(self.plan,self.parent,self.p/kind)
            self.assertEqual(r['status'],'STOPPED' if kind=='reject' else 'INTERRUPTED')
            s=self.read(kind)
            for key in ('model','adam','ssb','rng','frozen','history','ssb_steps'):
                self.assertTrue(equal(s[key],self.base[key]),key)
    def test_wrong_hash(self):
        plan=copy.deepcopy(self.plan);plan['parent_sha256']='0'*64
        with self.assertRaises(AssertionError):c.run(plan,self.parent,self.p/'bad')
        self.assertFalse((self.p/'bad').exists())

if __name__=='__main__':unittest.main()

