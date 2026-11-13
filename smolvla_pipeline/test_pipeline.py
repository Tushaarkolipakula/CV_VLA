"""Focused tests for curriculum data integrity and padded chunk loss."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import h5py
import numpy as np
import torch

from .data import ChunkData
from .model import reduce_valid_loss
from .train import curriculum


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        with h5py.File(self.root/'pick_bowl_demo.hdf5','w') as f:
            data=f.create_group('data');data.attrs['problem_info']=json.dumps({'language_instruction':'pick bowl'})
            for name,kind,verified in [('demo_0','clean',False),('demo_1','clean',False),
                                       ('recovery_0','recovery',True),('recovery_1','recovery',False)]:
                g=data.create_group(name);g.attrs['kind']=kind
                if kind=='recovery':
                    g.attrs['source_demo']='demo_0';g.attrs['noise_level']=1
                    g.attrs['task_success']=verified;g.attrs['trigger_type']='contact'
                g.create_dataset('actions',data=np.arange(35,dtype=np.float32).reshape(5,7))
                o=g.create_group('obs')
                for key,width in [('gripper_states',2),('ee_pos',3),('ee_ori',3)]:o.create_dataset(key,data=np.ones((5,width)))
                for key in ['agentview_rgb','eye_in_hand_rgb']:o.create_dataset(key,data=np.zeros((5,4,4,3),dtype=np.uint8))
        self.ds=ChunkData([self.root],chunk_size=3)

    def tearDown(self):self.ds.close();self.temp.cleanup()

    def test_clean_alignment_and_terminal_padding(self):
        i=next(i for i,d in enumerate(self.ds.demos) if d['kind']=='clean')
        sample=self.ds.sample(i,3)
        np.testing.assert_array_equal(sample['action'][0],np.arange(28,35))
        self.assertEqual(sample['valid'].tolist(),[True,False,False])

    def test_recovery_alignment_and_reject_unverified(self):
        recovery=[i for i,d in enumerate(self.ds.demos) if d['kind']=='recovery']
        self.assertEqual(len(recovery),1)
        np.testing.assert_array_equal(self.ds.sample(recovery[0],0)['action'][0],np.arange(7))

    def test_source_split_no_leakage(self):
        groups=lambda ids:{(self.ds.demos[i]['task'],self.ds.demos[i]['source']) for i in ids}
        self.assertFalse(groups(self.ds.train)&groups(self.ds.val))

    def test_loss_ignores_padding_and_unused_action_dimensions(self):
        loss=torch.ones((1,3,32),requires_grad=True)
        valid=torch.tensor([[True,False,False]])
        value=reduce_valid_loss(loss,valid,7)
        self.assertEqual(value.item(),1.0);value.backward()
        self.assertEqual(loss.grad[:,1:].abs().sum().item(),0)
        self.assertEqual(loss.grad[:,:,7:].abs().sum().item(),0)

    def test_curriculum_clean_warmup_and_level_bounds(self):
        self.assertEqual(curriculum(0,100,'curriculum',5),(0,0))
        self.assertEqual(curriculum(99,100,'curriculum',5),(5,.3))
        self.assertEqual(curriculum(99,100,'clean',5),(0,0))

    def test_recovery_availability_does_not_change_task_balance(self):
        other=self.root/'pick_cup_demo.hdf5'
        shutil.copyfile(self.root/'pick_bowl_demo.hdf5',other)
        with h5py.File(other,'r+') as f:
            f['data'].attrs['problem_info']=json.dumps({'language_instruction':'pick cup'})
            del f['data/recovery_0'];del f['data/recovery_1']
        ds=ChunkData([self.root],chunk_size=3)
        # Exercise the complete pool explicitly; this test concerns sampling,
        # while source-split isolation is tested separately above.
        ds.train=list(range(len(ds.demos)))
        batch=ds.batch(np.random.default_rng(5),400,1,1.0)
        bowls=sum(t=='pick bowl' for t in batch['task'])
        self.assertTrue(150<bowls<250)
        self.assertEqual(int(batch['is_recovery'].sum()),bowls)
        ds.close()


if __name__=='__main__':unittest.main()
