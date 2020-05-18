#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import khf,kuhf

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'Si 0.,  0.,  0.; Si 1.3467560987,  1.3467560987,  1.3467560987'
    cell.a = '''0.            2.6935121974    2.6935121974
                2.6935121974  0.              2.6935121974
                2.6935121974  2.6935121974    0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh

    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell

cell = make_primitive_cell([9]*3)

class KnownValues(unittest.TestCase):
    def test_krhf_gamma_center(self):
        kpts = cell.make_kpts([3,3,3], with_gamma_point=True,point_group=True,time_reversal=True)
        kmf = khf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        self.assertAlmostEqual(kmf.e_tot, -7.56744279355879, 8)

    def test_krhf_monkhorst(self):
        kpts = cell.make_kpts([3,3,3], with_gamma_point=False,point_group=True,time_reversal=True)
        kmf = khf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        self.assertAlmostEqual(kmf.e_tot, -7.569565546524837, 8)

    def test_kuhf_gamma_center(self):
        cell.spin = 0
        kpts = cell.make_kpts([3,3,3], with_gamma_point=True,point_group=True,time_reversal=True)
        kumf = kuhf.KUHF(cell, kpts=kpts)
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, -7.567442788439032, 8)

    def test_kuhf_monkhorst(self):
        cell.spin = 0
        kpts = cell.make_kpts([3,3,3], with_gamma_point=False,point_group=True,time_reversal=True)
        kumf = kuhf.KUHF(cell, kpts=kpts)
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, -7.569565542783276, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf with k-point symmetry")
    unittest.main()
