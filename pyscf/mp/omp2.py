import numpy as np
from pyscf import lib, ao2mo
from pyscf.mp import ump2
from pyscf.cc import uccsd
from functools import reduce

def gen_g_hop(mp, mo_coeff, mo_occ):

    make_gen_fock(mp)

def make_gen_fock(mp):
    '''
    F_{ai}
    '''
    dm1 = mp.make_rdm1()
    dm2aa, dm2ab, dm2bb = mp.make_rdm2()

    nmoa, nmob = mp.nmo
    nocca, noccb = mp.nocc
    nvira, nvirb = nmoa - nocca, nmob - noccb
    fa = np.dot(mp.get_h1e()[0][nocca:,:], dm1[0][:,:nocca])
    fb = np.dot(mp.get_h1e()[1][noccb:,:], dm1[1][:,:noccb])


    return [fa,fb]

def get_h1e(mp, mo_coeff=None):
    if mo_coeff is None: mo_coeff = mp.mo_coeff
    hcore = mp.get_hcore()
    if mp.frozen is not 0 or mp.frozen is not None:
        raise NotImplementedError()
    h1ea = reduce(np.dot, (mo_coeff[0].T.conj(), hcore, mo_coeff[0]))
    h1eb = reduce(np.dot, (mo_coeff[1].T.conj(), hcore, mo_coeff[1]))
    return [h1ea, h1eb]


def make_rdm2_ppoo_ovvo(mp, t2=None):
    if t2 is None: t2 = mp.t2
    t2aa, t2ab, t2bb = t2

    nmoa, nmob = mp.get_nmo()
    nocca, noccb = mp.get_nocc()
    nvira, nvirb = nmoa - nocca, nmob - noccb

    dm1a, dm1b = mp.make_rdm1(t2)
    dm1a[np.diag_indices(nocca)] -= 1
    dm1b[np.diag_indices(noccb)] -= 1

    dm2_ppoo = np.zeros((nmoa,nmoa,nocca,nocca),dtype=t2aa.dtype)
    dm2_ovvo = np.zeros((nocca,nvira,nvira,nocca),dtype=t2aa.dtype)
    for i in range(nocca):
        dm2_ppoo[i,i,:,:] += dm1a.T[:nocca,:nocca]
        dm2_ppoo[:,:,i,i] += dm1a.T
        dm2_ppoo[:,i,i,:] -= dm1a.T[:,:nocca]
        dm2_ppoo[i,:,:,i] -= dm1a[:,:nocca]
        dm2_ovvo[i,:,:,i] -= dm1a[nocca:,nocca:]
    for i in range(nocca):
        for j in range(nocca):
            dm2_ppoo[i,i,j,j] += 1
            dm2_ppoo[i,j,j,i] -= 1

    dm2_PPOO = np.zeros((nmob,nmob,noccb,noccb),dtype=t2bb.dtype)
    dm2_OVVO = np.zeros((noccb,nvirb,nvirb,noccb),dtype=t2bb.dtype)
    for i in range(noccb):
        dm2_PPOO[i,i,:,:] += dm1b.T[:noccb,:noccb]
        dm2_PPOO[:,:,i,i] += dm1b.T
        dm2_PPOO[:,i,i,:] -= dm1b.T[:,:noccb]
        dm2_PPOO[i,:,:,i] -= dm1b[:,:noccb]
        dm2_OVVO[i,:,:,i] -= dm1b[noccb:,noccb:]
    for i in range(noccb):
        for j in range(noccb):
            dm2_PPOO[i,i,j,j] += 1
            dm2_PPOO[i,j,j,i] -= 1

    dm2_ppOO = np.zeros((nmoa,nmoa,noccb,noccb),dtype=t2ab.dtype)
    for i in range(nocca):
        dm2_ppOO[i,i,:,:] += dm1b.T[:noccb,:noccb]
    for i in range(noccb):
        dm2_ppOO[:,:,i,i] += dm1a.T
    for i in range(nocca):
        for j in range(noccb):
            dm2_ppOO[i,i,j,j] += 1

    return dm2_ppoo,dm2_ppOO,dm2_PPOO,dm2_ovvo,dm2_OVVO

def make_rdm2_vovo(mp, t2=None):
    if t2 is None: t2 = mp.t2
    t2aa, t2ab, t2bb = t2

    if not (mp.frozen is 0 or mp.frozen is None):
        raise NotImplementedError
    else:
        dm2_vovo = t2aa.transpose(2,0,3,1).conj()       
        dm2_VOVO = t2bb.transpose(2,0,3,1).conj()
        dm2_voVO = t2ab.transpose(2,0,3,1).conj()
    return dm2_vovo, dm2_voVO, dm2_VOVO



def kernel(mp):

    pass

class OUMP2(ump2.UMP2):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, t2=None):
        ump2.UMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if not (self.frozen is 0 or self.frozen is None):
            raise NotImplementedError("Frozen orbitals not supported yet for OMP2.")
       
        self.t2 = t2 
        self.eris = self.ao2mo(mo_coeff)
        make_rdm2_ppoo_ovvo(self)
        make_rdm2_vovo(self)

    def get_hcore(self):
        return self._scf.get_hcore()


    def ao2mo(self, mo_coeff=None):
        return _make_eris_outcore(self, mo_coeff)

    get_h1e = get_h1e


class _ChemistsERIs(uccsd._ChemistsERIs):
    '''(pq|rs)'''
    def __init__(self, mol=None):
        uccsd._ChemistsERIs.__init__(self, mol)

def _make_eris_outcore(mycc, mo_coeff=None):

    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]

    eris.feri = lib.H5TmpFile()
    eris.ooov = eris.feri.create_dataset('ooov', (nocca,nocca,nocca,nvira), 'f8')
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocca,nvira,nvira*(nvira+1)//2), 'f8')

    eris.OOOV = eris.feri.create_dataset('OOOV', (noccb,noccb,noccb,nvirb), 'f8')
    eris.OVVV = eris.feri.create_dataset('OVVV', (noccb,nvirb,nvirb*(nvirb+1)//2), 'f8')

    eris.ovOO = eris.feri.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8')
    eris.ovVV = eris.feri.create_dataset('ovVV', (nocca,nvira,nvirb*(nvirb+1)//2), 'f8')

    eris.OOvv = eris.feri.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8')
    eris.OVoo = eris.feri.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8')
    eris.OVvv = eris.feri.create_dataset('OVvv', (noccb,nvirb,nvira*(nvira+1)//2), 'f8')

    mol = mycc.mol
    tmpf = lib.H5TmpFile()
    if nocca > 0:
        ao2mo.general(mol, (orboa,moa,moa,moa), tmpf, 'aa')
        buf = np.empty((nmoa,nmoa,nmoa))
        for i in range(nocca):
            lib.unpack_tril(tmpf['aa'][i*nmoa:(i+1)*nmoa], out=buf)
            eris.ooov[i] = buf[:nocca,:nocca,nocca:]
            eris.ovvv[i] = lib.pack_tril(buf[nocca:,nocca:,nocca:])
        del(tmpf['aa'])

    if noccb > 0:
        ao2mo.general(mol, (orbob,mob,mob,mob), tmpf, 'bb')
        buf = np.empty((nmob,nmob,nmob))
        for i in range(noccb):
            lib.unpack_tril(tmpf['bb'][i*nmob:(i+1)*nmob], out=buf)
            eris.OOOV[i] = buf[:noccb,:noccb,noccb:]
            eris.OVVV[i] = lib.pack_tril(buf[noccb:,noccb:,noccb:])
        del(tmpf['bb'])

    if nocca > 0:
        ao2mo.general(mol, (orboa,moa,mob,mob), tmpf, 'ab')
        buf = np.empty((nmoa,nmob,nmob))
        for i in range(nocca):
            lib.unpack_tril(tmpf['ab'][i*nmoa:(i+1)*nmoa], out=buf)
            eris.ovOO[i] = buf[nocca:,:noccb,:noccb]
            eris.ovVV[i] = lib.pack_tril(buf[nocca:,noccb:,noccb:])

    if noccb > 0:
        ao2mo.general(mol, (orbob,mob,moa,moa), tmpf, 'ba')
        buf = np.empty((nmob,nmoa,nmoa))
        for i in range(noccb):
            lib.unpack_tril(tmpf['ba'][i*nmob:(i+1)*nmob], out=buf)
            eris.OOvv[i] = buf[:noccb,nocca:,nocca:]
            eris.OVoo[i] = buf[noccb:,:nocca,:nocca]
            eris.OVvv[i] = lib.pack_tril(buf[noccb:,nocca:,nocca:])

    return eris


