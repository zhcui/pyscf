#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Zhi-Hao Cui <zhcui0408@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft

class KnownValues(unittest.TestCase):
    def test_KUKSpU(self):
        cell = pgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''

        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 7
        cell.build()
        kmesh = [2, 1, 1]
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        U_idx = ["1 C 2p"]
        U_val = [5.0]

        mf = pdft.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, C_ao_lo='minao',
                minao_ref='gth-szv')
        mf.conv_tol = 1e-10
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -10.694460059491741, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.dft.kukspu")
    unittest.main()
