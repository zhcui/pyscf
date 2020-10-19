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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.dft import krks,kuks

L = 2.
He = pbcgto.Cell()
He.verbose = 0
He.a = np.eye(3)*L
He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
He.build()

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
    cell.spin = 0
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.build()
    return cell

cell = make_primitive_cell([17]*3)

def tearDownModule():
    global cell, He
    del cell, He

class KnownValues(unittest.TestCase):
    def test_krks_gamma_center(self):
        nk = [2,2,2]
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = krks.KRKS(cell, kpts=kpts0)
        kmf0.xc = 'lda'
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = krks.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_krks_monkhorst(self):
        nk = [2,2,2]
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(cell, kpts=kpts0)
        kmf0.xc = 'lda'
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = krks.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_kuks_gamma_center(self):
        nk = [2,2,2]
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kumf0 = kuks.KUKS(cell, kpts=kpts0)
        kumf0.xc = 'lda'
        kumf0 = pscf.addons.smearing_(kumf0, sigma=0.001, method='fermi',fix_spin=True)
        kumf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = kuks.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kumf0.e_tot, 7)

    def test_kuks_monkhorst(self):
        nk = [2,2,2]
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kumf0 = kuks.KUKS(cell, kpts=kpts0)
        kumf0.xc = 'lda'
        kumf0 = pscf.addons.smearing_(kumf0, sigma=0.001, method='fermi',fix_spin=True)
        kumf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = kuks.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kumf0.e_tot, 7)

    def test_rsh(self):
        nk = [2,2,2]
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0)
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = krks.KRKS(He, kpts=kpts)
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 9)

    def test_rsh_df(self):
        nk = [2,2,2]
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).density_fit()
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = krks.KRKS(He, kpts=kpts).density_fit()
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 9)

    def test_rsh_mdf(self):
        nk = [1,2,2]
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).mix_density_fit()
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = krks.KRKS(He, kpts=kpts).mix_density_fit()
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 9)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf with k-point symmetry")
    unittest.main()
