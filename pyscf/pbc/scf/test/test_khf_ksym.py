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
from pyscf.pbc.scf import khf,kuhf

L = 2.
He = pbcgto.Cell()
He.verbose = 0
He.a = np.eye(3)*L
He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
He.build()

def make_primitive_cell(mesh, spin=0):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'Si 0.,  0.,  0.; Si 1.3467560987,  1.3467560987,  1.3467560987'
    cell.a = '''0.            2.6935121974    2.6935121974
                2.6935121974  0.              2.6935121974
                2.6935121974  2.6935121974    0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh
    cell.spin = spin
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()
    return cell

cell = make_primitive_cell([17]*3)
nk = [1,2,2]

def tearDownModule():
    global cell, He, nk
    del cell, He, nk

class KnownValues(unittest.TestCase):
    def test_krhf_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = khf.KRHF(cell, kpts=kpts0).run()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(cell, kpts=kpts).run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_krhf_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = khf.KRHF(cell, kpts=kpts0).run()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(cell, kpts=kpts).run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = kuhf.KUHF(cell, kpts=kpts0)
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUHF(cell, kpts=kpts)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = kuhf.KUHF(cell, kpts=kpts0)
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUHF(cell, kpts=kpts)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_smearing(self):
        cell = make_primitive_cell([17,]*3, 8)
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = pscf.KUHF(cell, kpts=kpts0)
        kmf0 = pscf.addons.smearing_(kmf0, sigma=0.001, method='fermi', fix_spin=True)
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True, time_reversal_symmetry=True)
        kumf = pscf.KUHF(cell, kpts=kpts)
        kumf = pscf.addons.smearing_(kumf, sigma=0.001, method='fermi', fix_spin=True)
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kmf0.e_tot, 7)

    def test_krhf_df(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = khf.KRHF(He, kpts=kpts0).density_fit().run()
        
        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(He, kpts=kpts).density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_krhf_mdf(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = khf.KRHF(He, kpts=kpts0).mix_density_fit().run()

        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(He, kpts=kpts).mix_density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_df(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = kuhf.KUHF(He, kpts=kpts0).density_fit().run()

        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KUHF(He, kpts=kpts).density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_kuhf_mdf(self):
        kpts0 = He.make_kpts(nk)
        kmf0 = kuhf.KUHF(He, kpts=kpts0).mix_density_fit().run()

        kpts = He.make_kpts(nk, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KUHF(He, kpts=kpts).mix_density_fit().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)

    def test_init_guess_from_chkfile(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(cell, kpts=kpts)
        kmf.chkfile='test.chk'
        e_tot = kmf.kernel()

        kmf.init_guess = 'chk'
        dm = kmf.from_chk('test.chk')
        kmf.max_cycle = 1
        kmf.kernel(dm)
        self.assertAlmostEqual(kmf.e_tot, e_tot, 9)

        mo_coeff = pscf.chkfile.load('test.chk', 'scf/mo_coeff')
        mo_occ = pscf.chkfile.load('test.chk', 'scf/mo_occ')
        dm = kmf.make_rdm1(mo_coeff, mo_occ)
        kmf.max_cycle = 1
        kmf.kernel(dm)
        self.assertAlmostEqual(kmf.e_tot, e_tot, 9)

        kmf.__dict__.update(pscf.chkfile.load('test.chk', 'scf'))
        kmf.max_cycle = 1
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, e_tot, 9)

        kmf = pscf.KUHF(cell, kpts=kpts)
        kmf.chkfile='test.chk'
        e_tot = kmf.kernel()

        kmf.init_guess = 'chk'
        dm = kmf.from_chk('test.chk')
        kmf.max_cycle = 1
        kmf.kernel(dm)
        self.assertAlmostEqual(kmf.e_tot, e_tot, 9)

    def test_to_uhf(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRHF(cell, kpts=kpts)
        kmf.kernel()
        dm = kmf.make_rdm1()
        dm = np.asarray([dm,dm]) / 2.

        kumf = kmf.to_uhf()
        kumf.max_cycle = 1
        kumf.kernel(dm)
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 9)

    def test_to_rhf(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUHF(cell, kpts=kpts)
        kumf.kernel()
        dm = kumf.make_rdm1()

        kmf = kumf.to_rhf()
        kmf.max_cycle = 1
        kmf.kernel(dm[0]+dm[1])
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 9)

    def test_convert_from(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUHF(cell, kpts=kpts)
        kumf.kernel()
        dm = kumf.make_rdm1()

        kmf = pscf.KRHF(cell, kpts=kpts)
        kmf = kmf.convert_from_(kumf)
        kmf.max_cycle = 1
        kmf.kernel(dm[0]+dm[1])
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 9)

        dm = kmf.make_rdm1()
        kumf = kumf.convert_from_(kmf)
        kumf.max_cycle = 1
        kumf.kernel(np.asarray([dm,dm]) / 2.)
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 9)


if __name__ == '__main__':
    print("Full Tests for HF with k-point symmetry")
    unittest.main()
