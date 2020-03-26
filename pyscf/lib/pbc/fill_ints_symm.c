#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "pbc/optimizer.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))


int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

void sort3c_kks1(double complex *out, double *bufr, double *bufi,
                        int *kptij_idx, int *shls_slice, int *ao_loc,
                        int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                        int msh0, int msh1);

void sort3c_kks2_igtj(double complex *out, double *bufr, double *bufi,
                             int *kptij_idx, int *shls_slice, int *ao_loc,
                             int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                             int msh0, int msh1);


static void kron(double *c, double *a, double *b, int nrowa, int ncola, int nrowb, int ncolb)
{
    //a has dimension of [nrowa, ncola] with lda = nrowa
    //b has dimension of [nrowb, ncolb] with ldb = nrowb
    //c has dimension of [nrowa*nrowb, ncola*ncolb] with ldc = nrowa*nrowb
    int i,j;
    int One = 1;
    double D1 = 1.;
    for (i=0; i<nrowa*nrowb*ncola*ncolb; i++) {
        c[i] = 0;
    }
    double *pc = c, *pa, *pb;
    for (i=0; i<ncola; i++){
        pa = a + (size_t)nrowa*i;
        for (j=0; j<ncolb; j++){
            pb = b + (size_t)nrowb*j;
            dger_(&nrowb, &nrowa, &D1, pb, &One, pa, &One, pc, &nrowb);
            pc += nrowa*nrowb;
        }
    }
}

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
        env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
        env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
        env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

static int get_shltrip_idx(int *shlcen, int *shltrip_cen_idx, int nshltrip)
{
    int idx = -1;
    for (int i=0; i<nshltrip; i++)
    {
        if (shlcen[0] == shltrip_cen_idx[0+i*3] &&
            shlcen[1] == shltrip_cen_idx[1+i*3] &&
            shlcen[2] == shltrip_cen_idx[2+i*3]){
            idx = i;
            break;
        }
    }
    assert(idx >= 0);
    return idx;
}

static void multiply_Dmats(double* Tkji, double *Dmats, int *rot_loc, int rot_mat_size, int nop, int iop, 
                           int di, int dj, int dk, int l_i, int l_j, int l_k)
{
    int dkj = dk*dj;
    int dkji = dkj*di;
    double *pDmats, *pTk, *pTj, *pTi;
    double *Tkj = malloc(sizeof(double)*dkj*dkj);
    pDmats = Dmats+(size_t)rot_mat_size*iop;
    pTk = pDmats + rot_loc[l_k];
    pTj = pDmats + rot_loc[l_j];
    pTi = pDmats + rot_loc[l_i];
    kron(Tkj, pTk, pTj, dk, dk, dj, dj);
    kron(Tkji+(size_t)dkji*dkji*iop, Tkj, pTi, dkj, dkj, di, di);
    free(Tkj);
}

static void _nr3c_fill_symm_kk(int (*intor)(), void (*fsort)(),
                          double complex *out, int nkpts_ij,
                          int nkpts, int comp, int nimgs, int ish, int jsh,
                          double *buf, double *env_loc, double *Ls,
                          double *expkL_r, double *expkL_i, int *kptij_idx,
                          int *shls_slice, int *ao_loc,
                          CINTOpt *cintopt, PBCOpt *pbcopt,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                          int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size)
{
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];

    const char TRANS_N = 'N';
    const double D0 = 0;
    const double D1 = 1;
    const double ND1 = -1;
    const int One = 1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int l_i = bas[ANG_OF+ish*BAS_SLOTS];
    int l_j = bas[ANG_OF+jsh*BAS_SLOTS];
    int l_k;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    int dkmax;

    int i, dijm, dijmc, dijmk, empty;
    int ksh, iL0, iL, jL, iLcount; 
    int idx_L2, iop;

    int shls[3], shlcen[3];
    double *bufkk_r, *bufkk_i, *bufkL_r, *bufkL_i, *bufL, *pbuf, *cache;
    int (*fprescreen)();
    if (pbcopt != NULL) {
        fprescreen = pbcopt->fprescreen;
    } else {
        fprescreen = PBCnoscreen;
    }

    shls[0] = ish;
    shls[1] = jsh;
    shlcen[2] = shlcen_atm_idx[jsh];
    shlcen[1] = shlcen_atm_idx[ish];
    for (ksh = ksh0; ksh < ksh1; ksh++) {//loop over shells one by one
        l_k = bas[ANG_OF+ksh*BAS_SLOTS];
        dkmax = ao_loc[ksh+1] - ao_loc[ksh];
        dijm = dij * dkmax;
        dijmc = dijm * comp;
        dijmk = dijmc * nkpts;
        bufkk_r = buf;
        bufkk_i = bufkk_r + (size_t)nkpts * dijmk;
        bufkL_r = bufkk_i + (size_t)nkpts * dijmk;
        bufkL_i = bufkL_r + (size_t)MIN(nimgs,IMGBLK) * dijmk;
        bufL    = bufkL_i + (size_t)MIN(nimgs,IMGBLK) * dijmk;
        cache   = bufL    + (size_t)nimgs * dijmc;
        for (i = 0; i < nkpts*dijmk*OF_CMPLX; i++) {
            bufkk_r[i] = 0;
        }

        shls[2] = ksh;
        bool *int_flags_L2 = malloc(sizeof(bool)*nimgs*nimgs);
        for (i = 0; i < nimgs*nimgs; i++) {
            int_flags_L2[i] = false;
        }
        double *int_ijk_buf = malloc(sizeof(double)*dijmc*nimgs*nimgs);
        double *pint_ijk;

        shlcen[0] = shlcen_atm_idx[ksh];
        int L2iL_off = get_shltrip_idx(shlcen, shltrip_cen_idx, nshltrip) * nimgs * nimgs;
        int *pL2iL = L2iL + (size_t)L2iL_off;
        int *piop = ops + (size_t)L2iL_off;

        int dkj = dkmax*dj;
        int dkji = dkj * di;
        double *Tkji = malloc(sizeof(double)*dkji*dkji*nop);
        bool op_flags[nop];
        for (i = 0; i < nop; i++) {
            op_flags[i] = false;
        }

        for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
            iLcount = MIN(IMGBLK, nimgs - iL0);
            for (iL = iL0; iL < iL0+iLcount; iL++) {
                //shift_bas(env_loc, env, Ls, iptrxyz, iL);
                pbuf = bufL;
                for (jL = 0; jL < nimgs; jL++) {
                    shift_bas(env_loc, env, Ls, iptrxyz, iL);
                    shift_bas(env_loc, env, Ls, jptrxyz, jL);
                    if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                        idx_L2 = pL2iL[iL * nimgs + jL];
                        pint_ijk = int_ijk_buf+(size_t)dijmc*idx_L2;
                        iop = piop[iL * nimgs + jL];
                        if (int_flags_L2[idx_L2] == false){//build integral
                            int_flags_L2[idx_L2] = true;
                            int iL_irr = idx_L2 / nimgs;
                            int jL_irr = idx_L2 % nimgs;
                            shift_bas(env_loc, env, Ls, iptrxyz, iL_irr);
                            shift_bas(env_loc, env, Ls, jptrxyz, jL_irr);
                            if ((*intor)(pint_ijk, NULL, shls, atm, natm, bas, nbas,
                                         env_loc, cintopt, cache)) {
                                empty = 0;
                            }
                        }
                        //multiply Wigner D matrices
                        if (op_flags[iop] == false) {
                            multiply_Dmats(Tkji, Dmats, rot_loc, rot_mat_size, nop, iop,
                                           di, dj, dkmax, l_i, l_j, l_k);
                            op_flags[iop] = true;
                        }
                        dgemm_(&TRANS_N, &TRANS_N, &dkji, &One, &dkji,
                               &D1, Tkji+(size_t)dkji*dkji*iop, &dkji, pint_ijk, &dkji,
                               &D0, pbuf, &dkji);
                    } else {
                        for (i = 0; i < dijmc; i++) {
                            pbuf[i] = 0;
                        }
                    }
                    pbuf += dijmc;
                }
                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
                       &D1, bufL, &dijmc, expkL_r, &nimgs,
                       &D0, bufkL_r+(iL-iL0)*(size_t)dijmk, &dijmc);
                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
                       &D1, bufL, &dijmc, expkL_i, &nimgs,
                       &D0, bufkL_i+(iL-iL0)*(size_t)dijmk, &dijmc);
            } // iL in range(0, nimgs)
            // conj(exp(1j*dot(h,k)))
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_r, &dijmk, expkL_r+iL0, &nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_i, &dijmk, expkL_i+iL0, &nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &D1, bufkL_i, &dijmk, expkL_r+iL0, &nimgs,
                   &D1, bufkk_i, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                   &ND1, bufkL_r, &dijmk, expkL_i+iL0, &nimgs,
                   &D1, bufkk_i, &dijmk);
        }
        free(Tkji);
        free(int_flags_L2);
        free(int_ijk_buf);
        (*fsort)(out, bufkk_r, bufkk_i, kptij_idx, shls_slice,
                 ao_loc, nkpts, nkpts_ij, comp, ish, jsh,
                 ksh, ksh+1);
    }
}

void PBCnr3c_fill_symm_kks2(int (*intor)(), double complex *out, int nkpts_ij,
                       int nkpts, int comp, int nimgs, int ish, int jsh,
                       double *buf, double *env_loc, double *Ls,
                       double *expkL_r, double *expkL_i, int *kptij_idx,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, PBCOpt *pbcopt,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                       int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size)
{
    int ip = ish + shls_slice[0];
    int jp = jsh + shls_slice[2] - nbas;
    if (ip > jp) {
        _nr3c_fill_symm_kk(intor, &sort3c_kks2_igtj, out,
                      nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                      buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                      shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                      shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                      L2iL, ops, Dmats, rot_loc, nop, rot_mat_size);
    } else if (ip == jp) {
        _nr3c_fill_symm_kk(intor, &sort3c_kks1, out,
                      nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                      buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                      shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                      shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                      L2iL, ops, Dmats, rot_loc, nop, rot_mat_size);
    }
}

void PBCnr3c_symm_drv(int (*intor)(), void (*fill)(), double complex *eri,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv,
                 int *shlcen_atm_idx, int *shltrip_cen_idx, int nshltrip,
                 int *L2iL, int *ops, double *Dmats, int *rot_loc, int nop, int rot_mat_size)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
    double *expkL_i = expkL_r + nimgs*nkpts;
    int i;
    for (i = 0; i < nimgs*nkpts; i++) {
        expkL_r[i] = creal(expkL[i]);
        expkL_i[i] = cimag(expkL[i]);
    }

    assert(comp == 1);
    size_t count;
    if (fill == &PBCnr3c_fill_symm_kks2) {
        int dijk =(GTOmax_shell_dim(ao_loc, shls_slice+0, 1) *
                   GTOmax_shell_dim(ao_loc, shls_slice+2, 1) *
                   GTOmax_shell_dim(ao_loc, shls_slice+4, 1));
        count = nkpts*nkpts * OF_CMPLX +
                nkpts*MIN(nimgs,IMGBLK) * OF_CMPLX + nimgs;
        //MAX(INTBUFMAX, dijk) to ensure buffer is enough for at least one (i,j,k) shell
        count*= MAX(INTBUFMAX, dijk) * comp;
    } else {
        count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
        count+= nimgs * nkpts * OF_CMPLX;
    }
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas, env);

    #pragma omp parallel
    {
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        memcpy(env_loc, env, sizeof(double)*nenv);
        double *buf = malloc(sizeof(double)*(count+cache_size));
        #pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
            ish = ij / njsh;
            jsh = ij % njsh;
            (*fill)(intor, eri, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                    buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                    shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env,
                    shlcen_atm_idx, shltrip_cen_idx, nshltrip,
                    L2iL, ops, Dmats, rot_loc, nop, rot_mat_size);
        }
        free(buf);
        free(env_loc);
    }
    free(expkL_r);
}
