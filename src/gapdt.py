##
## Circuitscape (C) 2008, Brad McRae and Viral B. Shah. 
##
## $Id: gapdt.py 740 2011-04-13 02:50:14Z viral $
##

from numpy import *
from scipy import sparse

class gapdt:

    def sprand (self, x):
        s = sparse.coo_matrix(x);

    def sprand (self, m, n, density):        
        nnz = fix(m*n*density)
        i = random.random_integers(0, m-1, nnz)
        j = random.random_integers(0, n-1, nnz)
        ij = c_[i, j].T
        data = random.rand(nnz)
        
        s = sparse.coo_matrix((data, ij), shape=(m, n))
        return s

    def relabel(self, oldlabel, offset=0):
        newlabel = zeros(size(oldlabel), dtype='int32')
        s = sort(oldlabel)
        perm = argsort(oldlabel)
        f = where(diff(concatenate(([s[0]-1], s))))
        newlabel[f] = 1
        newlabel = cumsum(newlabel)
        newlabel[perm] = copy(newlabel)
        return newlabel-1+offset

    def subsref(self, A, I, J):
        B = A[:, J][I, :]
        
        return B

    def deleterowcol(self, A, delrow, delcol):
        m = A.shape[0]
        n = A.shape[1]

        keeprows = delete (arange(0, m), delrow)
        keepcols = delete (arange(0, n), delcol)

        return A[keeprows][:,keepcols]

    def conditional_hooking (self, D, star, u, v):
        Du = D[u]
        Dv = D[v]
        
        hook = where ((star[u] == 1) & (Du > Dv))
        D[Du[hook]] = Dv[hook]

        return D

    def unconditional_hooking (self, D, star, u, v):
        Du = D[u]
        Dv = D[v]
        
        hook = where((star[u] == 1) & (Du != Dv))
        D[Du[hook]] = Dv[hook]

        return D

    def check_stars (self, D, star):
        star[:] = 1
        notstars = where (D != D[D])
        star[notstars] = 0
        star[D[D[notstars]]] = 0
        star = star[D]
        return star

    def components(self, G):
        Gcoo = sparse.coo_matrix(G)
        n = G.shape[0]
        U = Gcoo.row
        V = Gcoo.col

        D = arange (0, n, dtype='int32')
        star = zeros(n, 'int32')

        all_stars = False
        while True:
            star = self.check_stars (D, star)
            D = self.conditional_hooking(D, star, U, V)

            star = self.check_stars (D, star)
            D = self.unconditional_hooking (D, star, U, V)

            # pointer jumping
            D = D[D]

            if all_stars == True:
                return self.relabel(D, 1)
            
            if sum(star) == n:
                all_stars = True
