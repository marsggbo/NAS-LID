import copy
import math
import random
import numpy as np
import skdim
import torch
import torch.nn.functional as F
from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutator.random_mutator import RandomMutator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FewshotMutator(RandomMutator):
    def __init__(
        self,
        model,
        to_sample_similar: bool=False,
        *args, **kwargs
    ):
        '''
        '''
        super(FewshotMutator, self).__init__(model)
        self.to_sample_similar = to_sample_similar

    def sample_search(self, *args, **kwargs):
        if len(self._cache) > 0:
            result = self._cache
        else:
            result = super().sample_search(*args, **kwargs)

        supernet_mask = getattr(self, 'supernet_mask', None)
        if self.to_sample_similar:
            result = self.sample_similar(result, supernet_mask)
        elif supernet_mask is not None:
            supernet_mask = supernet_mask
            for key, val in supernet_mask.items():
                new_op_encoding = self.random_mutate_op(key, result[key], val)
                result[key] = new_op_encoding
        return result

    def sample_final(self):
        return super().sample_search()

    def sample_similar(self, mask, supernet_mask=None):
        similar_mask = copy.deepcopy(mask)
        keys = list(mask.keys())
        key = random.choice(keys)
        if supernet_mask is not None:
            new_op_encoding = self.random_mutate_op(key, mask[key], supernet_mask[key])
        else:
            new_op_encoding = self.random_mutate_op(key, mask[key])
        similar_mask[key] = new_op_encoding
        return similar_mask

    def random_mutate_op(self, op_key: str, op_encoding: torch.Tensor, supernet_op_encoding=None):
        '''
        Args:
            constraint: 
                0: no constraint
                1: only mutate the op_encoding where the mask is 1 when given the supernet_mask
        '''
        mutable = self[op_key]
        new_op_encoding = op_encoding.clone()
        if not getattr(mutable, 'is_freeze', False):
            # 1. mutable shoule be not frozen
            # 2. mutable's mask should be not all ones
            #    e.g., [1,0,1] indicates the second operation is disabled,
            #    so we should sample only the first or third operation.
            candidate_indices = torch.where(op_encoding==0)[0].cpu()
            if supernet_op_encoding is not None:
                candidate_indices2 = torch.where(supernet_op_encoding==1)[0]
                candidate_indices = torch.tensor([i for i in candidate_indices if i in candidate_indices2])
            gen_index = random.choice(candidate_indices)
            new_op_encoding = F.one_hot(gen_index, num_classes=len(op_encoding)).view(-1).bool()
        return new_op_encoding


if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network

    def test(net, fm):
        print('\n========test========')
        for i in range(60):
            if fm.idx == fm.num_path:
                print('new round\n=====================')
            fm.reset()
            print(f"{i+1}: {net.arch_encoding} {fm.idx}")
