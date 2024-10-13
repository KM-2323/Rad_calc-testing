from rad_settings_opt import *

import scipy.sparse.linalg as sp
import scipy.linalg as linalg
import util as util
import numpy as np


def spin(ndocc,norbs,cis_coeffs,nstates,cis_option,hetero):
    spinmat = np.zeros((nstates,nstates)) 
    if hetero=='no':
        #1 <0|S**2|0>
        spinmat[0,0] = 0.75
        #2 <0|S**2|ibar->0bar> = 0
        #3 <0|S**2|0->j'>  = 0
        #4 <0|S**2|i->j'>  = 0
        #5 <0|S**2|ibar->jbar'>  = 0  
        #6 <0|S**2|ibar->0bar,0->j'>  = 0
        #7 <ibar->0bar|S**2|kbar->0bar> only non-zero if i==k 
        for i in range(ndocc):
            spinmat[i+1,i+1] = 0.75
        #8 <kbar->0bar|S**2|0->j'> = 0      
        #9 <kbar->0bar|S**2|i->j'> = 0     
        #10 <kbar->0bar|S**2|ibar->jbar'> = 0 
        #11 <kbar->0bar|S**2|ibar->0bar,0->j'> = 0 
        #12 <0->l'|S**2|0->j'>
        for j in range(ndocc):
            spinmat[j+ndocc+1,j+ndocc+1] = 0.75
        #13 <0->l'|S**2|i->j'> = 0
        #14 <0->l'|S**2|ibar->jbar'> = 0  
        #15 <0->l'|S**2|ibar->0bar,0->j'> = 0          
        #16 <k->l'|S**2|i->j'>
        for m in range(ndocc**2):
            mm = m+2*ndocc+1
            spinmat[mm,mm] = 7/4   
        #17 <k->l'|S**2|ibar->jbar'> 
        for m in range(ndocc**2):
            mm = m+2*ndocc+1
            nn = m+ndocc**2+2*ndocc+1
            spinmat[mm,nn] = -1
            spinmat[nn,mm] = spinmat[mm,nn]
        #18 <k->l'|S**2|ibar->0bar,0->j'> 
        if cis_option == 'cisd':
            for m in range(ndocc**2):
                mm = m+2*ndocc+1
                nn = m+2*ndocc**2+2*ndocc+1
                spinmat[mm,nn] = 1
                spinmat[nn,mm] = spinmat[mm,nn]
        #19 <kbar->lbar'|S**2|ibar->jbar'>
        for m in range(ndocc**2):
            mm = m+ndocc**2+2*ndocc+1
            spinmat[mm,mm] = 7/4 
        #20 <kbar->lbar'|S**2|ibar->0bar,0->j'> 
        if cis_option == 'cisd':
            for m in range(ndocc**2):
                mm = m+ndocc**2+2*ndocc+1
                nn = m+2*ndocc**2+2*ndocc+1
                spinmat[mm,nn] = -1 
                spinmat[nn,mm] = spinmat[mm,nn]
        #21 <kbar->0bar,0->l'|S**2|ibar->0bar,0->j'>
        if cis_option == 'cisd':
            for m in range(ndocc**2):
                mm = m+2*ndocc**2+2*ndocc+1
                spinmat[mm,mm] = 7/4
    if hetero=='yes':
        nunocc = norbs-ndocc-1
        #1 <0|S**2|0>
        spinmat[0,0] = 0.75
        #2 <0|S**2|ibar->0bar> = 0
        #3 <0|S**2|0->j'>  = 0
        #4 <0|S**2|i->j'>  = 0
        #5 <0|S**2|ibar->jbar'>  = 0  
        #6 <0|S**2|ibar->0bar,0->j'>  = 0
        #7 <ibar->0bar|S**2|kbar->0bar> only non-zero if i==k
        for i in range(ndocc):
            spinmat[i+1,i+1] = 0.75
        #8 <kbar->0bar|S**2|0->j'> = 0      
        #9 <kbar->0bar|S**2|i->j'> = 0     
        #10 <kbar->0bar|S**2|ibar->jbar'> = 0 
        #11 <kbar->0bar|S**2|ibar->0bar,0->j'> = 0 
        #12 <0->l'|S**2|0->j'>
        for j in range(nunocc):
            spinmat[j+ndocc+1,j+ndocc+1] = 0.75
        #13 <0->l'|S**2|i->j'> = 0
        #14 <0->l'|S**2|ibar->jbar'> = 0  
        #15 <0->l'|S**2|ibar->0bar,0->j'> = 0  
        #16 <k->l'|S**2|i->j'>
        for m in range(ndocc*nunocc):
            mm = m+ndocc+nunocc+1
            spinmat[mm,mm] = 7/4
        #17 <k->l'|S**2|ibar->jbar'> 
        for m in range(ndocc*nunocc):
            mm = m+ndocc+nunocc+1
            nn = m+ndocc+nunocc+ndocc*nunocc+1
            spinmat[mm,nn] = -1
            spinmat[nn,mm] = spinmat[mm,nn]   
        #18 <k->l'|S**2|ibar->0bar,0->j'> 
        for m in range(ndocc*nunocc):
            mm = m+ndocc+nunocc+1
            nn = m+ndocc+nunocc+2*ndocc*nunocc+1
            spinmat[mm,nn] = 1
            spinmat[nn,mm] = spinmat[mm,nn]
        #19 <kbar->lbar'|S**2|ibar->jbar'>  
        for m in range(ndocc*nunocc):
            mm = m+ndocc+nunocc+ndocc*nunocc+1
            spinmat[mm,mm] = 7/4 
        #20 <kbar->lbar'|S**2|ibar->0bar,0->j'> 
        for m in range(ndocc*nunocc):
            mm = m+ndocc+nunocc+ndocc*nunocc+1
            nn = m+ndocc+nunocc+2*ndocc*nunocc+1
            spinmat[mm,nn] = -1 
            spinmat[nn,mm] = spinmat[mm,nn]    
        #21 <kbar->0bar,0->l'|S**2|ibar->0bar,0->j'>
        for m in range(ndocc*nunocc): 
            mm = m+ndocc+nunocc+2*ndocc*nunocc+1
            spinmat[mm,mm] = 7/4          
    s2=np.dot(cis_coeffs.T,np.dot(spinmat,cis_coeffs))
    s4=np.dot(cis_coeffs.T,np.dot(spinmat,np.dot(spinmat,cis_coeffs)))
    deltassq = np.sqrt(np.abs(s4-s2**2))
    return s2, deltassq  

def dipole(coords,atoms,norbs,hforbs,ndocc,nstates,basis,cis_option,hetero):
    print("Calculating dipole moments ...\n")
    # Routine to calculate the one electron dipole moment matrix (x, y and z) 
    # in the basis of orbitals, and then the dipole moment matrix in the basis
    # of excitations from the many electron determinant
    natoms = coords.shape[0]
    o0 = ndocc   
    dip1el = np.zeros((norbs,norbs,3))
    for i in range(norbs):
        for j in range(i,norbs):
            for u in range(natoms):
                # for x in range(3):
                #     dip1el[i,j,x] += hforbs[u,i]*coords[u,x]*hforbs[u,j]*tobohr
                #     dip1el[j,i,x] = dip1el[i,j,x]
                dip1el[i,j,:] += hforbs[u,i]*coords[u,:]*hforbs[u,j]*tobohr
                dip1el[j,i,:] = dip1el[i,j,:]
   # print("Checking one electron dipole moment array is symmetric (a value of zero means matrix is symmetric) ...")
   # print("x norm= %f"%linalg.norm(dip1el[:,:,0] - dip1el[:,:,0].T))  # checking symmetric
   # print("y norm= %f"%linalg.norm(dip1el[:,:,1] - dip1el[:,:,1].T))
   # print("z norm= %f"%linalg.norm(dip1el[:,:,2] - dip1el[:,:,2].T))
   # print(" ")
    dipoles = np.zeros((nstates,nstates,3)) 
    if basis=='xct' and hetero=='no':
        #1 <0|mu|0>
        # for x in range(3):
        #     for m in range(ndocc):
        #         dipoles[0,0,x] -= 2*dip1el[m,m,x]
        #     dipoles[0,0,x] -= dip1el[o0,o0,x]
        
        for m in range(ndocc):
            dipoles[0,0,:] -= 2*dip1el[m,m,:]
        dipoles[0,0,:] -= dip1el[o0,o0,:]
        #2 <0|mu|ibar->0bar> 
        # for x in range(3):
        #     for i in range(ndocc):
        #         dipoles[0,i+1,x] = -dip1el[i,o0,x]
        #         dipoles[i+1,0,x] = dipoles[0,i+1,x] 
       
        for i in range(ndocc):
            dipoles[0,i+1,:] = -dip1el[i,o0,:]
            dipoles[i+1,0,:] = dipoles[0,i+1,:] 
        #3 <0|mu|0->j'>
        # for x in range(3):
        #     for j in range (ndocc):
        #         dipoles[0,j+ndocc+1,x] = -dip1el[o0,j+ndocc+1,x]
        #         dipoles[j+ndocc+1,0,x] = dipoles[0,j+ndocc+1,x]
        
        for j in range (ndocc):
            dipoles[0,j+ndocc+1,:] = -dip1el[o0,j+ndocc+1,:]
            dipoles[j+ndocc+1,0,:] = dipoles[0,j+ndocc+1,:]
        #4 <0|mu|i->j'> 
        # for x in range(3):
        #     for n in range (ndocc**2):
        #         nn = n+2*ndocc+1
        #         i = int(np.floor(n/ndocc))
        #         j = n-i*ndocc+ndocc +1
        #         dipoles[0,nn,x] = -dip1el[i,j,x]
        #         dipoles[nn,0,x] = dipoles[0,nn,x]
        for n in range (ndocc**2):
                nn = n+2*ndocc+1
                i = int(np.floor(n/ndocc))
                j = n-i*ndocc+ndocc +1
                dipoles[0,nn,:] = -dip1el[i,j,:]
                dipoles[nn,0,:] = dipoles[0,nn,:]
        # for x in range(3):
        for n in range (ndocc**2):
            nn = n+2*ndocc+1
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            dipoles[0,nn,:] = -dip1el[i,j,:]
            dipoles[nn,0,:] = dipoles[0,nn,:]
        #5 <0|mu|ibar->jbar'>
        # for x in range(3):
        #     for n in range (ndocc**2):
        #         nn = n+ndocc**2+2*ndocc+1
        #         i = int(np.floor(n/ndocc))
        #         j = n-i*ndocc+ndocc +1
        #         dipoles[0,nn,x] = -dip1el[i,j,x]
        #         dipoles[nn,0,x] = dipoles[0,nn,x]
        for n in range (ndocc**2):
                nn = n+ndocc**2+2*ndocc+1
                i = int(np.floor(n/ndocc))
                j = n-i*ndocc+ndocc +1
                dipoles[0,nn,:] = -dip1el[i,j,:]
                dipoles[nn,0,:] = dipoles[0,nn,:]
        #6 <0|mu|ibar->0bar,0->j'> = 0
        if mixing==True:
            print("Dipole moments are corrected for ground state mixing of excited configurations")
            #7 <kbar->0bar|mu|ibar->0bar> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for i in range(ndocc):
            #         for k in range(ndocc):
            #             dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
            #             if i==k:
            #                 dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[o0,o0,x]
            
            for i in range(ndocc):
                for k in range(ndocc):
                    dipoles[i+1,k+1,:] = +dip1el[i,k,:] 
                    if i==k:
                        dipoles[i+1,k+1,:] += dipoles[0,0,:] - dip1el[o0,o0,:]
                    
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|i->j'> = 0
            #10 <kbar->0bar|mu|ibar->jbar'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+ndocc**2+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[i+1,nn,x] = -dip1el[o0,j,x]
            #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]  
            for n in range(ndocc**2):
                    nn = n+ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    dipoles[i+1,nn,:] = -dip1el[o0,j,:]
                    dipoles[nn,i+1,:] = dipoles[i+1,nn,:]  
            #11 <kbar->0bar|mu|ibar->0bar,0->j'>
            if cis_option == 'cisd':
                # for x in range(3):
                #    for n in range(ndocc**2):
                #         nn = n+2*ndocc**2+2*ndocc+1
                #         i = int(np.floor(n/ndocc))
                #         j = n-i*ndocc+ndocc +1
                #         dipoles[i+1,nn,x] = -dip1el[o0,j,x]
                #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
                for n in range(ndocc**2):
                        nn = n+2*ndocc**2+2*ndocc+1
                        i = int(np.floor(n/ndocc))
                        j = n-i*ndocc+ndocc +1
                        dipoles[i+1,nn,:] = -dip1el[o0,j,:]
                        dipoles[nn,i+1,:] = dipoles[i+1,nn,:]
            #12 <0->l'|mu|0->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for j in range(ndocc):
            #         for l in range(ndocc):
            #             dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
            #             if j==l:
            #                dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[o0,o0,x]
            for j in range(ndocc):
                    for l in range(ndocc):
                        dipoles[j+ndocc+1,l+ndocc+1,:] = -dip1el[j+ndocc+1,l+ndocc+1,:]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,:] += dipoles[0,0,:] + dip1el[o0,o0,:]
            #13 <0->l'|mu|i->j'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[j,nn,x] = dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc**2):
                    nn = n+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    dipoles[j,nn,:] = dip1el[i,o0,:]
                    dipoles[nn,j,:] = dipoles[j,nn,:]
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[j,nn,x] = dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc**2):
                nn = n+2*ndocc+1
                i = int(np.floor(n/ndocc))
                j = n-i*ndocc+ndocc +1
                dipoles[j,nn,:] = dip1el[i,o0,:]
                dipoles[nn,j,:] = dipoles[j,nn,:]
            #14 <0->l'|mu|ibar->jbar'> = 0
            #15 <0->l'|mu|ibar->0bar,0->j'>
            if cis_option == 'cisd':
                # for x in range(3):
                #     for n in range(ndocc**2):
                #         nn = n+2*ndocc**2+2*ndocc+1
                #         i = int(np.floor(n/ndocc))
                #         j = n-i*ndocc+ndocc +1
                #         dipoles[j,nn,x] = -dip1el[i,o0,x]
                #         dipoles[nn,j,x] = dipoles[j,nn,x]
                for n in range(ndocc**2):
                        nn = n+2*ndocc**2+2*ndocc+1
                        i = int(np.floor(n/ndocc))
                        j = n-i*ndocc+ndocc +1
                        dipoles[j,nn,:] = -dip1el[i,o0,:]
                        dipoles[nn,j,:] = dipoles[j,nn,:]
            #16 <k->l'|mu|i->j'> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc**2):
            #          mm = m+2*ndocc+1
            #          k = int(np.floor(m/ndocc))
            #          l = m-k*ndocc+ndocc +1
            #          for n in range (ndocc**2):
            #              nn = n+2*ndocc+1
            #              i = int(np.floor(n/ndocc))
            #              j = n-i*ndocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc**2):
                     mm = m+2*ndocc+1
                     k = int(np.floor(m/ndocc))
                     l = m-k*ndocc+ndocc +1
                     for n in range (ndocc**2):
                         nn = n+2*ndocc+1
                         i = int(np.floor(n/ndocc))
                         j = n-i*ndocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]
            #17 <k->l'|mu|ibar->jbar'> = 0
            #18 <k->l'|mu|ibar->0bar,0->j'> = 0
            #19 <kbar->lbar'|mu|ibar->jbar'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for m in range(ndocc**2):
            #         mm = m+ndocc**2+2*ndocc+1
            #         k = int(np.floor(m/ndocc))
            #         l = m-k*ndocc+ndocc +1
            #         for n in range (ndocc**2):
            #             nn = n+ndocc**2+2*ndocc+1
            #             i = int(np.floor(n/ndocc))
            #             j = n-i*ndocc+ndocc +1
            #             if i==k:
            #                 dipoles[mm,nn,x] = -dip1el[j,l,x]
            #             if j==l:
            #                 dipoles[mm,nn,x] += dip1el[i,k,x]
            #             if i==k and j==l:
            #                 dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc**2):
                mm = m+ndocc**2+2*ndocc+1
                k = int(np.floor(m/ndocc))
                l = m-k*ndocc+ndocc +1
                for n in range (ndocc**2):
                    nn = n+ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    if i==k:
                        dipoles[mm,nn,:] = -dip1el[j,l,:]
                    if j==l:
                        dipoles[mm,nn,:] += dip1el[i,k,:]
                    if i==k and j==l:
                        dipoles[mm,nn,:] += dipoles[0,0,:]
            #20 <kbar->lbar'|mu|ibar->0bar,0->j'> = 0
            #21 <kbar->0bar,0->l'|mu|ibar->0bar,0->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            if cis_option == 'cisd':
                # for x in range(3):
                    # for m in range(ndocc**2):
                    #     mm = m+2*ndocc**2+2*ndocc+1
                    #     k = int(np.floor(m/ndocc))
                    #     l = m-k*ndocc+ndocc +1
                    #     for n in range (ndocc**2):
                    #         nn = n+2*ndocc**2+2*ndocc+1
                    #         i = int(np.floor(n/ndocc))
                    #         j = n-i*ndocc+ndocc +1
                    #         if i==k:
                    #             dipoles[mm,nn,x] = -dip1el[j,l,x]
                    #         if j==l:
                    #             dipoles[mm,nn,x] += dip1el[i,k,x]
                    #         if i==k and j==l:
                    #             dipoles[mm,nn,x] += dipoles[0,0,x]
                for m in range(ndocc**2):
                        mm = m+2*ndocc**2+2*ndocc+1
                        k = int(np.floor(m/ndocc))
                        l = m-k*ndocc+ndocc +1
                        for n in range (ndocc**2):
                            nn = n+2*ndocc**2+2*ndocc+1
                            i = int(np.floor(n/ndocc))
                            j = n-i*ndocc+ndocc +1
                            if i==k:
                                dipoles[mm,nn,:] = -dip1el[j,l,:]
                            if j==l:
                                dipoles[mm,nn,:] += dip1el[i,k,:]
                            if i==k and j==l:
                                dipoles[mm,nn,:] += dipoles[0,0,:]
            #print(linalg.norm(dipoles[:,:,0] - dipoles[:,:,0].T))  # checking symmetric
           # print(linalg.norm(dipoles[:,:,1] - dipoles[:,:,1].T))
           # print(linalg.norm(dipoles[:,:,2] - dipoles[:,:,2].T))
    if basis == 'rot' and hetero=='no':
        #1 <0|mu|0>
        # for x in range(3):
        #     for m in range(ndocc):
        #         dipoles[0,0,x] -= 2*dip1el[m,m,x]
        #     dipoles[0,0,x] -= dip1el[o0,o0,x]
        for m in range(ndocc):
                dipoles[0,0,:] -= 2*dip1el[m,m,:]
        dipoles[0,0,:] -= dip1el[o0,o0,:]
        #2 <0|mu|ibar->0bar> 
        # for x in range(3):
        #     for i in range(ndocc):
        #         dipoles[0,i+1,x] = -dip1el[i,o0,x]
        #         dipoles[i+1,0,x] = dipoles[0,i+1,x] 
        for i in range(ndocc):
            dipoles[0,i+1,:] = -dip1el[i,o0,:]
            dipoles[i+1,0,:] = dipoles[0,i+1,:] 
        #3 <0|mu|0->j'>
        # for x in range(3):
        #     for j in range (ndocc):
        #         dipoles[0,j+ndocc+1,x] = -dip1el[o0,j+ndocc+1,x]
        #         dipoles[j+ndocc+1,0,x] = dipoles[0,j+ndocc+1,x]
        for j in range (ndocc):
                dipoles[0,j+ndocc+1,:] = -dip1el[o0,j+ndocc+1,:]
                dipoles[j+ndocc+1,0,:] = dipoles[0,j+ndocc+1,:]
        #4 <0|mu|4,i->j'>=0
        #5 <0|mu|2S,i->j'>
        # for x in range(3):
        #     for n in range (ndocc**2):
        #         nn = n+ndocc**2+2*ndocc+1
        #         i = int(np.floor(n/ndocc))
        #         j = n-i*ndocc+ndocc +1
        #         dipoles[0,nn,x] = -np.sqrt(2)*dip1el[i,j,x]
        #         dipoles[nn,0,x] = dipoles[0,nn,x]
        for n in range (ndocc**2):
            nn = n+ndocc**2+2*ndocc+1
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            dipoles[0,nn,:] = -np.sqrt(2)*dip1el[i,j,:]
            dipoles[nn,0,:] = dipoles[0,nn,:]
        #6 <0|mu|2T,i->j'>=0
        if mixing==True:
            print("Dipole moments are corrected for ground state mixing of excited configurations in rotated basis.\n")
            #7 <kbar->0bar|mu|ibar->0bar>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for i in range(ndocc):
            #         for k in range(ndocc):
            #             dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
            #             if i==k:
            #                 dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[o0,o0,x]
            for i in range(ndocc):
                    for k in range(ndocc):
                        dipoles[i+1,k+1,:] = +dip1el[i,k,:] 
                        if i==k:
                            dipoles[i+1,k+1,:] += dipoles[0,0,:] - dip1el[o0,o0,:]
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|4,i->j'>=0
            #10 <kbar->0bar|mu|2S,i->j'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+ndocc**2+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[i+1,nn,x] = -1/np.sqrt(2)*dip1el[o0,j,x]
            #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            for n in range(ndocc**2):
                nn = n+ndocc**2+2*ndocc+1
                i = int(np.floor(n/ndocc))
                j = n-i*ndocc+ndocc +1
                dipoles[i+1,nn,:] = -1/np.sqrt(2)*dip1el[o0,j,:]
                dipoles[nn,i+1,:] = dipoles[i+1,nn,:]
            #11 <kbar->0bar|mu|2T,i->j'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+2*ndocc**2+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[i+1,nn,x] = -3/np.sqrt(6)*dip1el[o0,j,x]
            #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            for n in range(ndocc**2):
                    nn = n+2*ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    dipoles[i+1,nn,:] = -3/np.sqrt(6)*dip1el[o0,j,:]
                    dipoles[nn,i+1,:] = dipoles[i+1,nn,:]
            
            #12 <0->j'|mu|0->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for j in range(ndocc):
            #         for l in range(ndocc):
            #             dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
            #             if j==l:
            #                dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[o0,o0,x]
            for j in range(ndocc):
                    for l in range(ndocc):
                        dipoles[j+ndocc+1,l+ndocc+1,:] = -dip1el[j+ndocc+1,l+ndocc+1,:]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,:] += dipoles[0,0,:] + dip1el[o0,o0,:]
            #13 <0->j'|mu|4,k->l'> = 0
            #14 <0->j'|mu|2S,k->l'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+ndocc**2+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[j,nn,x] = 1/np.sqrt(2)*dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc**2):
                    nn = n+ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    dipoles[j,nn,:] = 1/np.sqrt(2)*dip1el[i,o0,:]
                    dipoles[nn,j,:] = dipoles[j,nn,:]
            #15 <0->j'|mu|2T,k->l'>
            # for x in range(3):
            #     for n in range(ndocc**2):
            #         nn = n+2*ndocc**2+2*ndocc+1
            #         i = int(np.floor(n/ndocc))
            #         j = n-i*ndocc+ndocc +1
            #         dipoles[j,nn,x] = -3/np.sqrt(6)*dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc**2):
                    nn = n+2*ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    dipoles[j,nn,:] = -3/np.sqrt(6)*dip1el[i,o0,:]
                    dipoles[nn,j,:] = dipoles[j,nn,:]
            #16 <4,i->j'|mu|4,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc**2):
            #          mm = m+2*ndocc+1
            #          k = int(np.floor(m/ndocc))
            #          l = m-k*ndocc+ndocc +1
            #          for n in range (ndocc**2):
            #              nn = n+2*ndocc+1
            #              i = int(np.floor(n/ndocc))
            #              j = n-i*ndocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc**2):
                     mm = m+2*ndocc+1
                     k = int(np.floor(m/ndocc))
                     l = m-k*ndocc+ndocc +1
                     for n in range (ndocc**2):
                         nn = n+2*ndocc+1
                         i = int(np.floor(n/ndocc))
                         j = n-i*ndocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]   
            #17 <4,k->l'|mu|2S,i->j'> = 0
            #18 <4,k->l'|mu|2T,i->j'> = 0
            #19 <2S,i->j'|mu|2S,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc**2):
            #          mm = m+ndocc**2+2*ndocc+1
            #          k = int(np.floor(m/ndocc))
            #          l = m-k*ndocc+ndocc +1
            #          for n in range (ndocc**2):
            #              nn = n+ndocc**2+2*ndocc+1
            #              i = int(np.floor(n/ndocc))
            #              j = n-i*ndocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc**2):
                mm = m+ndocc**2+2*ndocc+1
                k = int(np.floor(m/ndocc))
                l = m-k*ndocc+ndocc +1
                for n in range (ndocc**2):
                    nn = n+ndocc**2+2*ndocc+1
                    i = int(np.floor(n/ndocc))
                    j = n-i*ndocc+ndocc +1
                    if i==k:
                        dipoles[mm,nn,:] = -dip1el[j,l,:]
                    if j==l:
                        dipoles[mm,nn,:] += dip1el[i,k,:]
                    if i==k and j==l:
                        dipoles[mm,nn,:] += dipoles[0,0,:]
            #20 <2S,i->j'|mu|2T,k->l'> = 0 
            
            #21 <2T,i->j'|mu|2T,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc**2):
            #          mm = m+2*ndocc**2+2*ndocc+1
            #          k = int(np.floor(m/ndocc))
            #          l = m-k*ndocc+ndocc +1
            #          for n in range (ndocc**2):
            #              nn = n+2*ndocc**2+2*ndocc+1
            #              i = int(np.floor(n/ndocc))
            #              j = n-i*ndocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc**2):
                     mm = m+2*ndocc**2+2*ndocc+1
                     k = int(np.floor(m/ndocc))
                     l = m-k*ndocc+ndocc +1
                     for n in range (ndocc**2):
                         nn = n+2*ndocc**2+2*ndocc+1
                         i = int(np.floor(n/ndocc))
                         j = n-i*ndocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]                 
            #print("%10.5f"%linalg.norm(dipoles[:,:,0] - dipoles[:,:,0].T))  # checking symmetric
            #print("%10.5f"%linalg.norm(dipoles[:,:,1] - dipoles[:,:,1].T))
            #print("%10.5f"%linalg.norm(dipoles[:,:,2] - dipoles[:,:,2].T))   
            
    if basis=='xct' and hetero=='yes':
        nunocc = norbs-ndocc-1
        #1 <0|mu|0>
        for x in range(3):
            for m in range(ndocc):
                dipoles[0,0,x] -= 2*dip1el[m,m,x]
            dipoles[0,0,x] -= dip1el[o0,o0,x]
        #2 <0|mu|ibar->0bar> 
        for x in range(3):
            for i in range(ndocc):
                dipoles[0,i+1,x] = -dip1el[i,o0,x]
                dipoles[i+1,0,x] = dipoles[0,i+1,x] 
        #3 <0|mu|0->j'>
        for x in range(3):
            for j in range (nunocc):
                dipoles[0,j+ndocc+1,x] = -dip1el[o0,j+ndocc+1,x]
                dipoles[j+ndocc+1,0,x] = dipoles[0,j+ndocc+1,x]
        #4 <0|mu|i->j'> 
        for x in range(3):
            for n in range (ndocc*nunocc):
                nn = n+ndocc+nunocc+1
                i = int(np.floor(n/nunocc))
                j = n-i*nunocc+ndocc +1
                dipoles[0,nn,x] = -dip1el[i,j,x]
                dipoles[nn,0,x] = dipoles[0,nn,x]
        #5 <0|mu|ibar->jbar'>
        for x in range(3):
            for n in range (ndocc*nunocc):
                nn = n+ndocc+nunocc+ndocc*nunocc+1
                i = int(np.floor(n/nunocc))
                j = n-i*nunocc+ndocc +1
                dipoles[0,nn,x] = -dip1el[i,j,x]
                dipoles[nn,0,x] = dipoles[0,nn,x]
        #6 <0|mu|ibar->0bar,0->j'> = 0
        if mixing==True:
            print("Dipole moments are corrected for ground state mixing of excited configurations")
            #7 <kbar->0bar|mu|ibar->0bar>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            for x in range(3):
                for i in range(ndocc):
                    for k in range(ndocc):
                        dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
                        if i==k:
                            dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[o0,o0,x]
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|i->j'> = 0
            #10 <kbar->0bar|mu|ibar->jbar'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -dip1el[o0,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]  
            #11 <kbar->0bar|mu|ibar->0bar,0->j'>
            for x in range(3):
               for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*(ndocc*nunocc)+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -dip1el[o0,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            #12 <0->l'|mu|0->j'> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            for x in range(3):
                for j in range(nunocc):
                    for l in range(nunocc):
                        dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[o0,o0,x]
            #13 <0->l'|mu|i->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = dip1el[i,o0,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #14 <0->l'|mu|ibar->jbar'> = 0
            #15 <0->l'|mu|ibar->0bar,0->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*(ndocc*nunocc)+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = -dip1el[i,o0,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #16 <k->l'|mu|i->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            for x in range(3):
                 for m in range(ndocc*nunocc):
                     mm = m+ndocc+nunocc+1
                     k = int(np.floor(m/nunocc))
                     l = m-k*nunocc+ndocc +1
                     for n in range (ndocc*nunocc):
                         nn = n+ndocc+nunocc+1
                         i = int(np.floor(n/nunocc))
                         j = n-i*nunocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,x] = -dip1el[j,l,x]
                         if j==l:
                             dipoles[mm,nn,x] += dip1el[i,k,x]
                         if i==k and j==l:
                             dipoles[mm,nn,x] += dipoles[0,0,x]
            #17 <k->l'|mu|ibar->jbar'> = 0
            #18 <k->l'|mu|ibar->0bar,0->j'> = 0
            #19 <kbar->lbar'|mu|ibar->jbar'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            for x in range(3):
                for m in range(ndocc*nunocc):
                    mm = m+ndocc+nunocc+ndocc*nunocc+1
                    k = int(np.floor(m/nunocc))
                    l = m-k*nunocc+ndocc +1
                    for n in range (ndocc*nunocc):
                        nn = n+ndocc+nunocc+ndocc*nunocc+1
                        i = int(np.floor(n/nunocc))
                        j = n-i*nunocc+ndocc +1
                        if i==k:
                            dipoles[mm,nn,x] = -dip1el[j,l,x]
                        if j==l:
                            dipoles[mm,nn,x] += dip1el[i,k,x]
                        if i==k and j==l:
                            dipoles[mm,nn,x] += dipoles[0,0,x]
            #20 <kbar->lbar'|mu|ibar->0bar,0->j'> = 0
            #21 <kbar->0bar,0->l'|mu|ibar->0bar,0->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            for x in range(3):
                for m in range(ndocc*nunocc):
                    mm = m+ndocc+nunocc+2*(ndocc*nunocc)+1
                    k = int(np.floor(m/nunocc))
                    l = m-k*nunocc+ndocc +1
                    for n in range (ndocc*nunocc):
                        nn = n+ndocc+nunocc+2*(ndocc*nunocc)+1
                        i = int(np.floor(n/nunocc))
                        j = n-i*nunocc+ndocc +1
                        if i==k:
                            dipoles[mm,nn,x] = -dip1el[j,l,x]
                        if j==l:
                            dipoles[mm,nn,x] += dip1el[i,k,x]
                        if i==k and j==l:
                            dipoles[mm,nn,x] += dipoles[0,0,x]
    
    if basis == 'rot' and hetero=='yes':
        nunocc = norbs-ndocc-1
        #1 <0|mu|0>
        # for x in range(3):
        #     for m in range(ndocc):
        #         dipoles[0,0,x] -= 2*dip1el[m,m,x]
        #     dipoles[0,0,x] -= dip1el[o0,o0,x]
        for m in range(ndocc):
            dipoles[0,0,:] -= 2*dip1el[m,m,:]
        dipoles[0,0,:] -= dip1el[o0,o0,:]
        #2 <0|mu|ibar->0bar> 
        # for x in range(3):
        #     for i in range(ndocc):
        #         dipoles[0,i+1,x] = -dip1el[i,o0,x]
        #         dipoles[i+1,0,x] = dipoles[0,i+1,x] 
        for i in range(ndocc):
                dipoles[0,i+1,:] = -dip1el[i,o0,:]
                dipoles[i+1,0,:] = dipoles[0,i+1,:] 
        #3 <0|mu|0->j'>
        # for x in range(3):
        #     for j in range (nunocc):
        #         dipoles[0,j+ndocc+1,x] = -dip1el[o0,j+ndocc+1,x]
        #         dipoles[j+ndocc+1,0,x] = dipoles[0,j+ndocc+1,x]
        for j in range (nunocc):
                dipoles[0,j+ndocc+1,:] = -dip1el[o0,j+ndocc+1,:]
                dipoles[j+ndocc+1,0,:] = dipoles[0,j+ndocc+1,:]
        #4 <0|mu|4,i->j'>=0
        #5 <0|mu|2S,i->j'>
        # for x in range(3):
        #     for n in range (ndocc*nunocc):
        #         nn = n+ndocc+nunocc+ndocc*nunocc+1
        #         i = int(np.floor(n/nunocc))
        #         j = n-i*nunocc+ndocc +1
        #         dipoles[0,nn,x] = -np.sqrt(2)*dip1el[i,j,x]
        #         dipoles[nn,0,x] = dipoles[0,nn,x]
        for n in range (ndocc*nunocc):
                nn = n+ndocc+nunocc+ndocc*nunocc+1
                i = int(np.floor(n/nunocc))
                j = n-i*nunocc+ndocc +1
                dipoles[0,nn,:] = -np.sqrt(2)*dip1el[i,j,:]
                dipoles[nn,0,:] = dipoles[0,nn,:]
        #6 <0|mu|2T,i->j'>=0
        if mixing==True:
            print("Dipole moments are corrected for ground state mixing of excited configurations in rotated basis.\n")
            #7 <kbar->0bar|mu|ibar->0bar>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for i in range(ndocc):
            #         for k in range(ndocc):
            #             dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
            #             if i==k:
            #                 dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[o0,o0,x]
            for i in range(ndocc):
                    for k in range(ndocc):
                        dipoles[i+1,k+1,:] = +dip1el[i,k,:] 
                        if i==k:
                            dipoles[i+1,k+1,:] += dipoles[0,0,:] - dip1el[o0,o0,:]
            
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|4,i->j'>=0
            #10 <kbar->0bar|mu|2S,i->j'>
            # for x in range(3):
            #     for n in range(ndocc*nunocc):
            #         nn = n+ndocc+nunocc+ndocc*nunocc+1
            #         i = int(np.floor(n/nunocc))
            #         j = n-i*nunocc+ndocc +1
            #         dipoles[i+1,nn,x] = -1/np.sqrt(2)*dip1el[o0,j,x]
            #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,:] = -1/np.sqrt(2)*dip1el[o0,j,:]
                    dipoles[nn,i+1,:] = dipoles[i+1,nn,:]
            #11 <kbar->0bar|mu|2T,i->j'>
            # for x in range(3):
            #     for n in range(ndocc*nunocc):
            #         nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            #         i = int(np.floor(n/nunocc))
            #         j = n-i*nunocc+ndocc +1
            #         dipoles[i+1,nn,x] = -3/np.sqrt(6)*dip1el[o0,j,x]
            #         dipoles[nn,i+1,x] = dipoles[i+1,nn,x]  
            for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,:] = -3/np.sqrt(6)*dip1el[o0,j,:]
                    dipoles[nn,i+1,:] = dipoles[i+1,nn,:]   
            #12 <0->j'|mu|0->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #     for j in range(nunocc):
            #         for l in range(nunocc):
            #             dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
            #             if j==l:
            #                dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[o0,o0,x]
            for j in range(nunocc):
                    for l in range(nunocc):
                        dipoles[j+ndocc+1,l+ndocc+1,:] = -dip1el[j+ndocc+1,l+ndocc+1,:]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,:] += dipoles[0,0,:] + dip1el[o0,o0,:]
            #13 <0->j'|mu|4,k->l'> = 0
            #14 <0->j'|mu|2S,k->l'>
            # for x in range(3):
            #     for n in range(ndocc*nunocc):
            #         nn = n+ndocc+nunocc+ndocc*nunocc+1
            #         i = int(np.floor(n/nunocc))
            #         j = n-i*nunocc+ndocc +1
            #         dipoles[j,nn,x] = 1/np.sqrt(2)*dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,:] = 1/np.sqrt(2)*dip1el[i,o0,:]
                    dipoles[nn,j,:] = dipoles[j,nn,:]
            #15 <0->j'|mu|2T,k->l'>
            # for x in range(3):
            #     for n in range(ndocc*nunocc):
            #         nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            #         i = int(np.floor(n/nunocc))
            #         j = n-i*nunocc+ndocc +1
            #         dipoles[j,nn,x] = -3/np.sqrt(6)*dip1el[i,o0,x]
            #         dipoles[nn,j,x] = dipoles[j,nn,x]
            for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,:] = -3/np.sqrt(6)*dip1el[i,o0,:]
                    dipoles[nn,j,:] = dipoles[j,nn,:]
            #16 <4,i->j'|mu|4,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc*nunocc):
            #          mm = m+ndocc+nunocc+1
            #          k = int(np.floor(m/nunocc))
            #          l = m-k*nunocc+ndocc +1
            #          for n in range (ndocc*nunocc):
            #              nn = n+ndocc+nunocc+1
            #              i = int(np.floor(n/nunocc))
            #              j = n-i*nunocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc*nunocc):
                     mm = m+ndocc+nunocc+1
                     k = int(np.floor(m/nunocc))
                     l = m-k*nunocc+ndocc +1
                     for n in range (ndocc*nunocc):
                         nn = n+ndocc+nunocc+1
                         i = int(np.floor(n/nunocc))
                         j = n-i*nunocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]                 
            #17 <4,k->l'|mu|2S,i->j'> = 0
            #18 <4,k->l'|mu|2T,i->j'> = 0
            #19 <2S,i->j'|mu|2S,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc*nunocc):
            #          mm = m+ndocc+nunocc+ndocc*nunocc+1
            #          k = int(np.floor(m/nunocc))
            #          l = m-k*nunocc+ndocc +1
            #          for n in range (ndocc*nunocc):
            #              nn = n+ndocc+nunocc+ndocc*nunocc+1
            #              i = int(np.floor(n/nunocc))
            #              j = n-i*nunocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc*nunocc):
                     mm = m+ndocc+nunocc+ndocc*nunocc+1
                     k = int(np.floor(m/nunocc))
                     l = m-k*nunocc+ndocc +1
                     for n in range (ndocc*nunocc):
                         nn = n+ndocc+nunocc+ndocc*nunocc+1
                         i = int(np.floor(n/nunocc))
                         j = n-i*nunocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]                
            #20 <2S,i->j'|mu|2T,k->l'> = 0 
            
            #21 <2T,i->j'|mu|2T,k->l'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
            # for x in range(3):
            #      for m in range(ndocc*nunocc):
            #          mm = m+ndocc+nunocc+2*ndocc*nunocc+1
            #          k = int(np.floor(m/nunocc))
            #          l = m-k*nunocc+ndocc +1
            #          for n in range (ndocc*nunocc):
            #              nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            #              i = int(np.floor(n/nunocc))
            #              j = n-i*nunocc+ndocc +1
            #              if i==k:
            #                  dipoles[mm,nn,x] = -dip1el[j,l,x]
            #              if j==l:
            #                  dipoles[mm,nn,x] += dip1el[i,k,x]
            #              if i==k and j==l:
            #                  dipoles[mm,nn,x] += dipoles[0,0,x]
            for m in range(ndocc*nunocc):
                     mm = m+ndocc+nunocc+2*ndocc*nunocc+1
                     k = int(np.floor(m/nunocc))
                     l = m-k*nunocc+ndocc +1
                     for n in range (ndocc*nunocc):
                         nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                         i = int(np.floor(n/nunocc))
                         j = n-i*nunocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,:] = -dip1el[j,l,:]
                         if j==l:
                             dipoles[mm,nn,:] += dip1el[i,k,:]
                         if i==k and j==l:
                             dipoles[mm,nn,:] += dipoles[0,0,:]
    perm_dip=dipoles[0,0,:]
    for i in range(natoms):
        atom_z=0
        if atoms[i][0] in ['C','c','n1','N1']:
            atom_z=1
        elif atoms[i][0] in ['Cl','cl','CL','N2','n2']:
            atom_z=2   
        # for x in range(3):
        #     perm_dip[x]+=atom_z*coords[i,x]*tobohr
        perm_dip[:]+=atom_z*coords[i,:]*tobohr
    print("Permanent dipole moment of ground state: mu0 = %7.3f x %7.3f y %7.3f z\n"%(perm_dip[0],perm_dip[1],perm_dip[2]))
    return dipoles
   
def transform(two_body,hf_orbs):
        #fock_mat_mo=np.dot(hf_orbs.T,np.dot(fock_mat,hf_orbs))
#place two-body terms into four index tensor----in site basis entire classes were zeroed out allowing storage in 2-D, 
#but this does not carry over into MO basis so prepare for this here
        two_body_4i=np.zeros((hf_orbs.shape[0],hf_orbs.shape[0],hf_orbs.shape[0],hf_orbs.shape[0]))
        for i in range (hf_orbs.shape[0]):
                for j in range (i,hf_orbs.shape[0]):
                        two_body_4i[i,i,j,j]=two_body[i,j]
                        two_body_4i[j,j,i,i]=two_body[i,j]
#four index transformation
        mat1=np.einsum("ij,klmi->klmj",hf_orbs,two_body_4i)
        mat2=np.einsum("ij,klim->kljm",hf_orbs,mat1)
        mat3=np.einsum("ij,kilm->kjlm",hf_orbs,mat2)
        two_body_mo=np.einsum("ij,iklm->jklm",hf_orbs,mat3)
        return two_body_mo

def broaden(FWHM,osc,energy):
    if brdn_typ == 'wavelength' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(%s/2))**2)" %(osc,evtonm/energy,FWHM)
    elif brdn_typ == 'energy' and line_typ == 'lorentzian':
        eqn="+%04.3f*1/(1+((%04.3f-x)/(0.5*%s*%04.3f*x))**2)"  %(osc,evtonm/energy,FWHM,evtonm/energy)
    elif brdn_typ == 'energy' and line_typ == 'gaussian':
        eqn="+%04.3f*exp(-((%04.3f-x)/(0.5*%s*%04.3f*x))**2)" %(osc,evtonm/energy,FWHM,evtonm/energy)
    return eqn


def cisd_ham_rot(ndocc,energy0,orb_energies,rep_tens):
    o0 = ndocc# no. of doubly-occupied orbitals
    nstates = 3*ndocc**2 +2*ndocc +1
    cish = np.zeros((nstates,nstates))     
    #1 <0|H|0>
    cish[0,0] = energy0
    #2 <0|H|ibar->0bar> 
    for i in range (ndocc): 
        cish[0,i+1] = 0.5*rep_tens[i,o0,o0,o0] #cish[0,i+1] = fock_mat_mo[i,o0] + 0.5*rep_tens[i,o0,o0,o0]
        cish[i+1,0] = cish[0,i+1]
    #3 <0|H|0->j'>
    for j in range (ndocc):
        cish[0,j+ndocc+1] = -0.5*rep_tens[o0,j+ndocc+1,o0,o0] #cish[0,j+ndocc+1] = fock_mat_mo[o0,j+ndocc+1] - 0.5*rep_tens[o0,j+ndocc+1,o0,o0]
        cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    #4 <0|H|Q i->j'> ALL ZERO
    
    #5 <0|H|D(+)i->j'> ALL ZERO (only depends on Fij')
        
    #6 <0|H|D(-)i->j'>
    for n in range (ndocc**2):
        nn = n+2*ndocc**2+2*ndocc+1
        i = int(np.floor(n/ndocc))
        j = n-i*ndocc+ndocc +1
        cish[0,nn] = 3/(np.sqrt(6)) *rep_tens[i,o0,o0,j]
        cish[nn,0] = cish[0,nn]
    #7 <kbar->0bar|H|ibar->0bar> 
    for i in range(ndocc):
        for k in range(i,ndocc):
            cish[i+1,k+1] = -rep_tens[i,k,o0,o0] +0.5*rep_tens[i,o0,o0,k]
            if i==k:
                cish[i+1,i+1] += energy0 + orb_energies[o0] - orb_energies[i] + 0.5*rep_tens[o0,o0,o0,o0]
            cish[k+1,i+1] = cish[i+1,k+1]
    #8 <kbar->0bar|H|0->j'>
    for j in range(ndocc):
        for k in range(ndocc):
            cish[k+1,j+ndocc+1] = rep_tens[o0,k,o0,j+ndocc+1]
            cish[j+ndocc+1,k+1] = cish[k+1,j+ndocc+1]
    #9 <0->l',+|H|Q i->j'> ALL ZERO
    
    #10 <kbar->0bar|H|D(+)i->j'>  
    for k in range(ndocc):
        for n in range (ndocc**2):
            nn = n+ndocc**2+2*ndocc+1
            i = int(np.floor(n/ndocc))
            jp = n-i*ndocc+ndocc +1
            cish[k+1,nn] = np.sqrt(2)*rep_tens[o0,k,i,jp] - 1/np.sqrt(2)*rep_tens[o0,jp,i,k]
            if i==k:
                cish[k+1,nn] += 1/(2*np.sqrt(2))*rep_tens[o0,jp,o0,o0]
            cish[nn,k+1] = cish[k+1,nn]
    #11 <kbar->0bar|H|D(-)i->j'>
    for k in range (ndocc):
        for n in range (ndocc**2):
            nn = n+2*ndocc**2+2*ndocc+1
            i = int(np.floor(n/ndocc))
            jp = n-i*ndocc+ndocc +1
            cish[k+1,nn] = -3/np.sqrt(6)*rep_tens[o0,jp,i,k]
            if i==k:
               cish[k+1,nn] += 3/(2*np.sqrt(6))*rep_tens[o0,jp,o0,o0] 
            cish[nn,k+1] = cish[k+1,nn]
    #12 <0->l'|H|0->j'>
    for j in range(ndocc):
        for l in range(j,ndocc):
            cish[j+ndocc+1,l+ndocc+1] = - rep_tens[j+ndocc+1,l+ndocc+1,o0,o0] + 0.5*rep_tens[j+ndocc+1,o0,o0,l+ndocc+1]
            if j==l:
                cish[j+ndocc+1,j+ndocc+1] += energy0 + orb_energies[j+ndocc+1] - orb_energies[o0] + 0.5*rep_tens[o0,o0,o0,o0]
            cish[l+ndocc+1,j+ndocc+1] = cish[j+ndocc+1,l+ndocc+1]
    #13 <0->l',-|H|Q i->j'> ALL ZERO
           
    #14 <0->l'|H|D(+)i->j'> 
    for lp in range(ndocc+1,2*ndocc+1):
        #print(lp)
        for n in range (ndocc**2):
            nn = n+ndocc**2+2*ndocc+1
            i = int(np.floor(n/ndocc))
            jp = n-i*ndocc+ndocc +1
            cish[lp,nn] = np.sqrt(2)*rep_tens[i,jp,lp,o0] -1/np.sqrt(2)*rep_tens[i,o0,lp,jp]
            if lp==jp:
                cish[lp,nn] += 1/(2*np.sqrt(2))*rep_tens[i,o0,o0,o0]
            cish[nn,lp] = cish[lp,nn]
    #15 <0->l'|H|D(-)i->j'>
    for lp in range(ndocc+1,2*ndocc+1):
        for n in range (ndocc**2):
            nn = n+2*ndocc**2+2*ndocc+1
            i = int(np.floor(n/ndocc))
            jp = n-i*ndocc+ndocc +1
            cish[lp,nn] = 3/np.sqrt(6)*rep_tens[i,o0,lp,jp]
            if jp==lp:
                cish[lp,nn] -= 3/(2*np.sqrt(6))*rep_tens[i,o0,o0,o0]
            cish[nn,lp] = cish[lp,nn]
    #16 <Qk->l'|H|Qi->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc**2):
        mm = m+2*ndocc+1
        k = int(np.floor(m/ndocc))
        l = m-k*ndocc+ndocc +1
        for n in range (m,ndocc**2):
            nn = n+2*ndocc+1
            #print(mm,nn)
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            cish[mm,nn] = -rep_tens[i,k,l,j]
            if i==k:
                cish[mm,nn] -= 0.5*rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] -= 0.5*rep_tens[o0,k,i,o0]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]
         
    #17 <Qk->l'|H|D(+)i->j'> ALL ZERO
            
    #18 <Qk->l'|H|D(-)i->j'> ALL ZERO
            
    #19 <D(+)i->j'|H|D(+)i->j'> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc**2):
        mm = m+ndocc**2+2*ndocc+1
        k = int(np.floor(m/ndocc))
        l = m-k*ndocc+ndocc +1
        for n in range (m,ndocc**2):
            nn = n+ndocc**2+2*ndocc+1
            #print(mm,nn)
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            cish[mm,nn] = 2*rep_tens[i,j,k,l] - rep_tens[i,k,j,l]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i] 
            cish[nn,mm] = cish[mm,nn]
            
    #20 <D(+)i->j'|H|D(-)i->j'> 
    for m in range(ndocc**2):
        mm = m+ndocc**2+2*ndocc+1
        k = int(np.floor(m/ndocc))
        l = m-k*ndocc+ndocc +1
        for n in range (ndocc**2):
            nn = n+2*ndocc**2+2*ndocc+1
            #print(mm,nn)
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            if i==k:
                cish[mm,nn] = 0.5*np.sqrt(3)*rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] -= 0.5*np.sqrt(3)*rep_tens[o0,k,i,o0]
            cish[nn,mm] = cish[mm,nn]
    
    #21 <D(-)i->j'|H|D(-)i->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc**2):
        mm = m+2*ndocc**2+2*ndocc+1
        k = int(np.floor(m/ndocc))
        l = m-k*ndocc+ndocc +1
        for n in range (m,ndocc**2):
            nn = n+2*ndocc**2+2*ndocc+1
            #print(mm,nn)
            i = int(np.floor(n/ndocc))
            j = n-i*ndocc+ndocc +1
            cish[mm,nn] = -rep_tens[i,k,l,j]
            if i==k:
                cish[mm,nn] += rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] += rep_tens[i,o0,o0,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]
    return cish

def cisd_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs):
    print("")
    print("------------------------")
    print("Starting ExROPPP calculation for monoradical in rotated basis")
    print("------------------------\n")
    # Transform 2-el ingrls into mo basis
    rep_tens = transform(repulsion,hf_orbs)
    # Construct CIS Hamiltonian
    ham_rot = cisd_ham_rot(ndocc,energy0,orb_energies,rep_tens)
    print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
    print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))
    o0 = ndocc
    nstates = 3*ndocc**2 +2*ndocc +1
    if states_cutoff_option == 'states' and states_to_print <= nstates:
        rng = states_to_print
        print('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
    else:
        rng = nstates
    if states_cutoff_option == 'energy':
        cutoff_energy = energy_cutoff
        print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
    else:
        cutoff_energy = 100
    # Diagonalize CIS Hamiltonianfor first rng excited states
    if rng<nstates:
        print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
        cis_energies,cis_coeffs=sp.eigsh(ham_rot,k=rng,which="SA")
    elif rng==nstates:
        print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
        cis_energies,cis_coeffs=linalg.eigh(ham_rot)
    dip_array = dipole(coords,atoms,norbs,hf_orbs,ndocc,nstates,'rot','cisd','no')
    aku=np.einsum("ijx,jk",dip_array,cis_coeffs)
    mu0u=np.einsum("j,jix",cis_coeffs[:,0].T,aku)
    osc_array=np.zeros_like(cis_energies)
    s2_array=np.zeros_like(cis_energies)
    print("Ground state energy relative to E(|0>): %04.3f eV"%(cis_energies[0]-energy0))
    rt = 2.**.5
    strng = ""
    for i in range(rng): # Loop over CIS states
        if cis_energies[i]-cis_energies[0] > cutoff_energy:
            break
        print("State %s %04.3f eV \n" % (i,cis_energies[i]-cis_energies[0])) #print("State %s %04.3f eV \n" % (i,energy-cis_energies[0]))
        print("Excitation    CI Coef    CI C*rt(2)")
        spin = 0 # initialise total spin
        for j in range (cis_coeffs.shape[0]): # Loop over configurations in each CIS state   
        # if configuration is the ground determinant
            if j == 0: 
                if np.absolute(cis_coeffs[j,i]) > 1e-2:
                    print('|0>           %10.5f'  %(cis_coeffs[j,i]))
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
                continue
        # if configuration is |ibar->0bar>     
            elif j>0 and j<=ndocc:
                iorb = j-1
                str1 = str(ndocc-iorb) + "bar" #str(iorb) + "bar" #
                str2 = "0bar" #"3bar"#
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
       # if configuration is |0->j'> 
            elif j>ndocc and j<=2*ndocc:
                jorb = j 
                str1 = "0" 
                str2 = str(jorb-ndocc)+"'" #str(jorb) #
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
        # if configuration is |Qi->j'>
            elif j>2*ndocc and j<=2*ndocc + ndocc**2:
                iorb = int(np.floor((j-2*ndocc-1)/ndocc))
                jorb = (j-2*ndocc-1)-iorb*ndocc+ndocc +1
                str1 = "Q " + str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'" 
                spin += 3.75*cis_coeffs[j,i]**2 # (S=1.5) 
         # if configuration is |D(S)i->j'> (bright doublet state)
            elif j>2*ndocc + ndocc**2 and j<=2*ndocc + 2*ndocc**2:
                iorb = int(np.floor((j-2*ndocc-ndocc**2-1)/ndocc))
                jorb = (j-2*ndocc-ndocc**2-1)-iorb*ndocc+ndocc +1
                str1 = "D(S) " +str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'"
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
         #if configuration is |D(T)i->j'> (dark doublet state)
            elif j>2*ndocc + 2*ndocc**2:
                iorb = int(np.floor((j-2*ndocc-2*ndocc**2-1)/ndocc))
                jorb = (j-2*ndocc-2*ndocc**2-1)-iorb*ndocc+ndocc +1
                str1 = "D(T) "+str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'"
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
            if np.absolute(cis_coeffs[j,i]) > 1e-1:
                print("%s->%s %10.5f %10.5f " \
                %(str1,str2,cis_coeffs[j,i],cis_coeffs[j,i]*rt))
        if i==0:
            print("\n<S**2>: %04.3f" %spin)
            print("--------------------------------------------------------------------\n")
            continue
        osc = 2.0/3.0*((cis_energies[i]-cis_energies[0])/toev)*(mu0u[i,0]**2+mu0u[i,1]**2+mu0u[i,2]**2) 
        osc_array[i]=osc
        s2_array[i]=spin
        print("")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.3f   <S**2>: %04.3f" % (mu0u[i,0], mu0u[i,1], mu0u[i,2], osc, spin))
        print("--------------------------------------------------------------------\n")
        #strng = strng + broaden(20.0,osc,cis_energies[i]-cis_energies[0]) 
        strng = strng + broaden(FWHM,osc,cis_energies[i]-cis_energies[0])
    strng = strng[1:]    
    return strng, cis_energies-cis_energies[0],osc_array,s2_array

def hetero_cisd_ham(ndocc,norbs,energy0,orb_energies,rep_tens):
    o0 = ndocc# no. of doubly-occupied orbitals
    nunocc = norbs-ndocc-1
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    cish = np.zeros((nstates,nstates))
    #cish[0:nelec,0:nelec] = cis_ham_sml
    #1 <0|H|0>
    cish[0,0] = energy0
    #2 <0|H|ibar->0bar> 
    for i in range (ndocc): 
        cish[0,i+1] = 0.5*rep_tens[i,o0,o0,o0] #cish[0,i+1] = fock_mat_mo[i,o0] + 0.5*rep_tens[i,o0,o0,o0]
        cish[i+1,0] = cish[0,i+1]
    #3 <0|H|0->j'>
    for j in range (nunocc):
        cish[0,j+ndocc+1] = -0.5*rep_tens[o0,j+ndocc+1,o0,o0] #cish[0,j+ndocc+1] = fock_mat_mo[o0,j+ndocc+1] - 0.5*rep_tens[o0,j+ndocc+1,o0,o0]
        cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    #4 <0|H|i->j'> # 
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = -0.5*rep_tens[i,o0,o0,j]
        cish[nn,0] = cish[0,nn]
    #5 <0|H|ibar->jbar'> #  
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = 0.5*rep_tens[i,o0,o0,j]
        cish[nn,0] = cish[0,nn]
    #6 <0|H|ibar->0bar,0->j'> #
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+2*ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = rep_tens[i,o0,o0,j]
        cish[nn,0] = cish[0,nn]
    #7 <kbar->0bar|H|ibar->0bar> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for i in range(ndocc):
        for k in range(ndocc):
            cish[i+1,k+1] = -rep_tens[i,k,o0,o0] +0.5*rep_tens[i,o0,o0,k]
            if i==k:
                cish[i+1,i+1] += energy0 + orb_energies[o0] - orb_energies[i] + 0.5*rep_tens[o0,o0,o0,o0]
            cish[k+1,i+1]=cish[i+1,k+1]
    #8 <kbar->0bar|H|0->j'>
    for j in range(nunocc):
        for k in range(ndocc):
            cish[k+1,j+ndocc+1] = rep_tens[o0,k,o0,j+ndocc+1]
            cish[j+ndocc+1,k+1] = cish[k+1,j+ndocc+1]
            
    #9 <kbar->0bar|H|i->j'> 
    for k in range(ndocc):    
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = rep_tens[o0,k,i,j]
            cish[nn,k+1] = cish[k+1 ,nn]
            
    #10 <kbar->0bar|H|ibar->jbar'> 
    for k in range(ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = rep_tens[o0,k,i,j] - rep_tens[o0,j,i,k] #10a) and 10b)
            if i==k:
                cish[k+1,nn] += 0.5*rep_tens[o0,j,o0,o0] #10a)
            cish[nn,k+1] = cish[k+1,nn]
            #
    #11 <kbar->0bar|H|ibar->0bar,0->j'> 
    for k in range (ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = -rep_tens[o0,j,i,k] #11a) and 11b)
            if i==k:
                cish[k+1,nn] += 0.5*rep_tens[o0,j,o0,o0] #11a)
            cish[nn,k+1] = cish[k+1,nn]
           
    #12 <0->l'|H|0->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for j in range(nunocc):
        for l in range(nunocc):
            cish[j+ndocc+1,l+ndocc+1] = - rep_tens[j+ndocc+1,l+ndocc+1,o0,o0] + 0.5*rep_tens[j+ndocc+1,o0,o0,l+ndocc+1]
            if j==l:
                cish[j+ndocc+1,j+ndocc+1] += energy0 + orb_energies[j+ndocc+1] - orb_energies[o0] + 0.5*rep_tens[o0,o0,o0,o0]

   # 13 <0->l'|H|i->j'>
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,j,l,o0] - rep_tens[i,o0,l,j]
            if l==j: 
                cish[l,nn] += 0.5*rep_tens[i,o0,o0,o0]
            cish[nn,l] = cish[l,nn]
           
    #14 <0->l'|H|ibar->jbar'>
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,j,l,o0]
            cish[nn,l] = cish[l,nn]
        
    #15 <0->l'|H|ibar->0bar,0->j'>  
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,o0,l,j]
            if l==j:
              cish[l,nn] += -0.5*rep_tens[i,o0,o0,o0] #15a)
            cish[nn,l] = cish[l,nn]
            
    #16 <k->l'|H|i->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = rep_tens[l,k,i,j] - rep_tens[l,j,i,k] #16d
            if i==k:
                cish[mm,nn] -= 0.5*rep_tens[l,o0,o0,j] #16b
            if j==l:
                cish[mm,nn] += 0.5*rep_tens[i,o0,o0,k] #16c
            if i==k and j==l:
                cish[mm,nn] += energy0 - orb_energies[i] + orb_energies[j] #16a)
         
    #17 <k->l'|H|ibar->jbar'> 
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = rep_tens[l,k,i,j]
            cish[nn,mm] = cish[mm,nn]
            
    #18 <k->l'|H|ibar->0bar,0->j'> 
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            if l==j:
                cish[mm,nn] = -rep_tens[o0,k,i,o0]
                cish[nn,mm] = cish[mm,nn]
            
    #19 <kbar->lbar'|H|ibar->jbar'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = rep_tens[l,k,i,j] - rep_tens[l,j,i,k]
            if i==k:
                cish[mm,nn] += 0.5*rep_tens[l,o0,o0,j] 
            if j==l:
                cish[mm,nn] -= 0.5*rep_tens[i,o0,o0,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 -orb_energies[i] + orb_energies[j]
            
    #20 <kbar->lbar'|H|ibar->0bar,0->j'> 
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            if i==k:
                cish[mm,nn] = rep_tens[l,o0,o0,j]
            cish[nn,mm] = cish[mm,nn]
        
    #21 <kbar->0bar,0->l'|H|ibar->0bar,0->j'> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc): 
        mm = m+ndocc+nunocc+2*ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = -rep_tens[i,k,l,j]
            if i==k:
                cish[mm,nn] += 0.5*rep_tens[l,o0,o0,j]
            if j==l:
                cish[mm,nn] += 0.5*rep_tens[i,o0,o0,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 - orb_energies[i] + orb_energies[j]
            
    return cish
            
def hetero_cisd(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs):
    print("")
    print("------------------------")
    print("Starting ExROPPP calculation for heterocycle monoradical in excitations basis")
    print("------------------------\n")
    # Transform 2-el ingrls into mo basis
    rep_tens = transform(repulsion,hf_orbs)
    print('Check repulsion tensor is symmetric')
    print(linalg.norm(rep_tens - rep_tens.T)) 
    # Construct and diagonalise CIS Hamiltonian for first 25 excited states
    cis_ham_het = hetero_cisd_ham(ndocc,norbs,energy0,orb_energies,rep_tens)
    # check it's symmetric
    print('Check Hamiltonian is symmetric')
    print(linalg.norm(cis_ham_het - cis_ham_het.T)) 
    #sys.exit()
    np.savetxt('big_ham_benz.csv', cis_ham_het, delimiter=',') 
    o0 = ndocc
    nunocc = norbs-ndocc-1
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    if states_cutoff_option == 'states' and states_to_print <= nstates:
        rng = states_to_print
        print('Lowest %d states. WARNING - Some states may not be included in spectrum.\n'%states_to_print)
    else:
        rng = nstates
    if states_cutoff_option == 'energy':
        cutoff_energy = energy_cutoff
        print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
    else:
        cutoff_energy = 100
    # Diagonalize CIS Hamiltonian for first rng excited states
    if rng<nstates:
        print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
        cis_energies,cis_coeffs=sp.eigsh(cis_ham_het,k=rng,which="SA")
    elif rng==nstates:
        print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
        cis_energies,cis_coeffs=linalg.eigh(cis_ham_het)
    # Calculate S**2 matrix
    s_squared,deltassq = spin(ndocc,norbs,cis_coeffs,nstates,'cisd','yes')
    #print("\nCheck spin mat is symmetric")
    #print(linalg.norm(s_squared - s_squared.T)) 
    # Calculate dipole moment array
    dip_array = dipole(coords,atoms,norbs,hf_orbs,ndocc,nstates,'xct','cisd','yes')
    aku=np.einsum("ijx,jk",dip_array,cis_coeffs)
    mu0u=np.einsum("j,jix",cis_coeffs[:,0].T,aku)
    rt = 2.**.5
    osc_array = np.zeros_like(cis_energies)
    print("--------------------------------\n")
    strng = ""
    # decide how many states to read off
    if states_cutoff_option == 'states' and states_to_print <= nstates:
        rng = states_to_print
        print('Lowest %d states. WARNING - Some states may not be included in spectrum.\n'%states_to_print)
    else:
        rng = nstates
    if states_cutoff_option == 'energy':
        cutoff_energy = energy_cutoff
        print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
    else:
        cutoff_energy = 100
    for i in range(rng): # Loop over CIS states
        if cis_energies[i]-cis_energies[0] > cutoff_energy:
            break
        print("State %s %04.3f eV \n" % (i,cis_energies[i]-cis_energies[0])) #print("State %s %04.3f eV \n" % (i,energy-cis_energies[0]))
        print("Excitation    CI Coef    CI C*rt(2)  tdmx    tdmy    tdmz")
        tot_tdmx = 0 # initialise total trans. dip. moment
        tot_tdmy = 0
        tot_tdmz = 0
        for j in range (cis_coeffs.shape[0]): # Loop over configurations in each CIS state   
        # if state is the ground state
            if j == 0 and np.absolute(cis_coeffs[j,i]) > 1e-2: 
                print('|0>           %10.5f'  %(cis_coeffs[j,i]))
                continue
        # if state is |ibar->0bar>     
            elif j>0 and j<=ndocc:
                tdmx = 0
                tdmy = 0
                tdmz = 0
                iorb = j-1
                str1 = str(ndocc-iorb) + "bar" #str(iorb) + "bar" #
                str2 = "0bar" #"3bar"#
                for k in range(norbs):
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,0]*tobohr*hf_orbs[k,o0] #
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,1]*tobohr*hf_orbs[k,o0]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,2]*tobohr*hf_orbs[k,o0] 
                tot_tdmx += tdmx
                tot_tdmy += tdmy
                tot_tdmz += tdmz
        # if state is |0->j'> 
            elif j>ndocc and j<=ndocc+nunocc:
                tdmx = 0
                tdmy = 0
                tdmz = 0
                jorb = j 
                str1 = "0" 
                str2 = str(jorb-ndocc)+"'" 
                #print(str2)
                for k in range(norbs):
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,o0]*coords[k,0]*tobohr*hf_orbs[k,jorb] 
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,o0]*coords[k,1]*tobohr*hf_orbs[k,jorb]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,o0]*coords[k,2]*tobohr*hf_orbs[k,jorb] 
                tot_tdmx += tdmx
                tot_tdmy += tdmy
                tot_tdmz += tdmz
        # if state is |i->j'>
            elif j>ndocc+nunocc and j<=ndocc+nunocc+ndocc*nunocc:
                tdmx = 0
                tdmy = 0
                tdmz = 0
                iorb = int(np.floor((j-ndocc-nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-1)-iorb*nunocc+ndocc +1
                #print(iorb,jorb)
                str1 = str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'" 
                for k in range(norbs):
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,0]*tobohr*hf_orbs[k,jorb] 
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,1]*tobohr*hf_orbs[k,jorb]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,2]*tobohr*hf_orbs[k,jorb] 
                tot_tdmx += tdmx
                tot_tdmy += tdmy
                tot_tdmz += tdmz

        # if state is |ibar->jbar'>
            elif j>ndocc+nunocc+ndocc*nunocc and j<=ndocc+nunocc+2*ndocc*nunocc:
                tdmx = 0
                tdmy = 0
                tdmz = 0
                iorb = int(np.floor((j-ndocc-nunocc-ndocc*nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                #print(iorb,jorb)
                str1 = str(ndocc-iorb)+"bar"
                str2 = str(jorb-ndocc)+"bar'" 
                for k in range(norbs):
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,0]*tobohr*hf_orbs[k,jorb] 
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,1]*tobohr*hf_orbs[k,jorb]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,2]*tobohr*hf_orbs[k,jorb] 
                tot_tdmx += tdmx
                tot_tdmy += tdmy
                tot_tdmz += tdmz
        #if state is |ibar->0bar,0->j'>
            elif j>ndocc+nunocc+2*ndocc*nunocc:
                tdmx = 0
                tdmy = 0
                tdmz = 0
                iorb = int(np.floor((j-ndocc-nunocc-2*ndocc*nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-2*ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                #print(iorb,jorb)
                str1 = str(ndocc-iorb)+"bar"
                str2 = "0bar, 0->" +str(jorb-ndocc)+"'" 
                
            if np.absolute(cis_coeffs[j,i]) > 10e-2:
                #print(j)
                print("%5s-> %5s %10.5f %10.5f %7.3f %7.3f %7.3f" \
                %(str1,str2,cis_coeffs[j,i],cis_coeffs[j,i]*rt,tdmx,tdmy,tdmz))
        if i==0:
            print("\n<S**2>: %04.3f   Delta(<S**2>): %04.3f" %(s_squared[i,i],deltassq[i,i]))
            continue
        osc = 2.0/3.0*((cis_energies[i]-cis_energies[0])/toev)*(mu0u[i,0]**2+mu0u[i,1]**2+mu0u[i,2]**2) 
        osc_array[i]=osc
        print("")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.3f   <S**2>: %04.3f   Delta(<S**2>): %04.3f" % (mu0u[i,0], mu0u[i,1], mu0u[i,2], osc, s_squared[i,i], deltassq[i,i]))
        print("--------------------------------------------------------------------\n")
        #strng = strng + broaden(20.0,osc,cis_energies[i]-cis_energies[0]) 
        strng = strng + broaden(FWHM,osc,cis_energies[i]-cis_energies[0])
    strng = strng[1:]
    return strng,cis_energies - cis_energies[0],osc_array

def hetero_ham_rot(ndocc,norbs,energy0,orb_energies,rep_tens):
    nunocc = norbs-ndocc-1
    o0=ndocc
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    cish = np.zeros((nstates,nstates))
    #1 <0|H|0>
    cish[0,0] = energy0
    #2 <0|H|ibar->0bar> 
    for i in range (ndocc): 
        cish[0,i+1] = 0.5*rep_tens[i,o0,o0,o0] 
        cish[i+1,0] = cish[0,i+1]
    #3 <0|H|0->j'>
    for j in range (nunocc):
        cish[0,j+ndocc+1] = -0.5*rep_tens[o0,j+ndocc+1,o0,o0] 
        cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    #4 <0|H|Q i->j'> ALL ZERO  
    #5 <0|H|D(+)i->j'> ALL ZERO (only depends on Fij')
    #6 <0|H|D(-)i->j'>
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+2*ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = 3/(np.sqrt(6)) *rep_tens[i,o0,o0,j]
        cish[nn,0] = cish[0,nn]
    #7 <kbar->0bar|H|ibar->0bar> 
    for i in range(ndocc):
        for k in range(i,ndocc):
            cish[i+1,k+1] = -rep_tens[i,k,o0,o0] +0.5*rep_tens[i,o0,o0,k]
            if i==k:
                cish[i+1,i+1] += energy0 + orb_energies[o0] - orb_energies[i] + 0.5*rep_tens[o0,o0,o0,o0]
            cish[k+1,i+1] = cish[i+1,k+1]
    #8 <kbar->0bar|H|0->j'>
    for j in range(nunocc):
        for k in range(ndocc):
            cish[k+1,j+ndocc+1] = rep_tens[o0,k,o0,j+ndocc+1]
            cish[j+ndocc+1,k+1] = cish[k+1,j+ndocc+1]
    #9 <0->l',+|H|Q i->j'> ALL ZERO
    #10 <kbar->0bar|H|D(+)i->j'>  
    for k in range(ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[k+1,nn] = np.sqrt(2)*rep_tens[o0,k,i,jp] - 1/np.sqrt(2)*rep_tens[o0,jp,i,k]
            if i==k:
                cish[k+1,nn] += 1/(2*np.sqrt(2))*rep_tens[o0,jp,o0,o0]
            cish[nn,k+1] = cish[k+1,nn]
    #11 <kbar->0bar|H|D(-)i->j'>
    for k in range (ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[k+1,nn] = -3/np.sqrt(6)*rep_tens[o0,jp,i,k]
            if i==k:
               cish[k+1,nn] += 3/(2*np.sqrt(6))*rep_tens[o0,jp,o0,o0] 
            cish[nn,k+1] = cish[k+1,nn]
    #12 <0->l'|H|0->j'>
    for j in range(nunocc):
        for l in range(j,nunocc):
            cish[j+ndocc+1,l+ndocc+1] = - rep_tens[j+ndocc+1,l+ndocc+1,o0,o0] + 0.5*rep_tens[j+ndocc+1,o0,o0,l+ndocc+1]
            if j==l:
                cish[j+ndocc+1,j+ndocc+1] += energy0 + orb_energies[j+ndocc+1] - orb_energies[o0] + 0.5*rep_tens[o0,o0,o0,o0]
            cish[l+ndocc+1,j+ndocc+1] = cish[j+ndocc+1,l+ndocc+1]
    #13 <0->l',-|H|Q i->j'> ALL ZERO         
    #14 <0->l'|H|D(+)i->j'> 
    for lp in range(ndocc+1,ndocc+nunocc+1):
        #print(lp)
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[lp,nn] = np.sqrt(2)*rep_tens[i,jp,lp,o0] -1/np.sqrt(2)*rep_tens[i,o0,lp,jp]
            if lp==jp:
                cish[lp,nn] += 1/(2*np.sqrt(2))*rep_tens[i,o0,o0,o0]
            cish[nn,lp] = cish[lp,nn]
    #15 <0->l'|H|D(-)i->j'>
    for lp in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[lp,nn] = 3/np.sqrt(6)*rep_tens[i,o0,lp,jp]
            if jp==lp:
                cish[lp,nn] -= 3/(2*np.sqrt(6))*rep_tens[i,o0,o0,o0]
            cish[nn,lp] = cish[lp,nn]
    #16 <Qk->l'|H|Qi->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (m,ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = -rep_tens[i,k,l,j]
            if i==k:
                cish[mm,nn] -= 0.5*rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] -= 0.5*rep_tens[o0,k,i,o0]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]        
    #17 <Qk->l'|H|D(+)i->j'> ALL ZERO           
    #18 <Qk->l'|H|D(-)i->j'> ALL ZERO
    #19 <D(+)k->l'|H|D(+)i->j'> CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (m,ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = 2*rep_tens[i,j,k,l] - rep_tens[i,k,j,l]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i] 
            cish[nn,mm] = cish[mm,nn]
    #20 <D(+)k->l'|H|D(-)i->j'> 
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            if i==k:
                cish[mm,nn] = 0.5*np.sqrt(3)*rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] -= 0.5*np.sqrt(3)*rep_tens[o0,k,i,o0]
            cish[nn,mm] = cish[mm,nn] 
    #21 <D(-)k->l'|H|D(-)i->j'>  CAN OPTIMISE TO RUN OVER UPPER TRIANGLE ONLY AND EQUATE LOWER TRAINGLE ELEMENTS TO THE TRANSPOSE OF UPPER TRAINGLE
    for m in range(ndocc*nunocc):
        mm = m+ndocc+nunocc+2*ndocc*nunocc+1
        k = int(np.floor(m/nunocc))
        l = m-k*nunocc+ndocc +1
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[mm,nn] = -rep_tens[i,k,l,j]
            if i==k:
                cish[mm,nn] += rep_tens[j,o0,o0,l]
            if j==l:
                cish[mm,nn] += rep_tens[i,o0,o0,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]
    return cish

def hetero_cisd_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs):
    print("")
    print("------------------------")
    print("Starting ExROPPP calculation for monoradical heterocycle in rotated basis")
    print("------------------------\n")
    # Transform 2-el ingrls into mo basis
    rep_tens = transform(repulsion,hf_orbs)
    # Construct CIS Hamiltonian
    het_ham_rot = hetero_ham_rot(ndocc,norbs,energy0,orb_energies,rep_tens)
    #print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
    #print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(het_ham_rot-het_ham_rot.T)))
    #print(linalg.norm(het_ham_rot-het_ham_rot.T))
    nunocc = norbs-ndocc-1
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    if states_cutoff_option == 'states' and states_to_print <= nstates:
        rng = states_to_print
        print('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
    else:
        rng = nstates
    if states_cutoff_option == 'energy':
        cutoff_energy = energy_cutoff
        print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
    else:
        cutoff_energy = 100
    # Diagonalize CIS Hamiltonianfor first rng excited states
    if rng<nstates:
        print("Diagonalizing Hamiltonian using the sparse matrix method ...\n")
        cis_energies,cis_coeffs=sp.eigsh(het_ham_rot,k=rng,which="SA")
    elif rng==nstates:
        print("Diagonalizing Hamiltonian using the dense matrix method ...\n")
        cis_energies,cis_coeffs=linalg.eigh(het_ham_rot)
    dip_array = dipole(coords,atoms,norbs,hf_orbs,ndocc,nstates,'rot','cisd','yes')
    aku=np.einsum("ijx,jk",dip_array,cis_coeffs)
    mu0u=np.einsum("j,jix",cis_coeffs[:,0].T,aku)
    osc_array=np.zeros_like(cis_energies)
    s2_array=np.zeros_like(cis_energies)
    print("Ground state energy relative to E(|0>): %04.3f eV"%(cis_energies[0]-energy0))
    rt = 2.**.5
    strng = ""
    eqn=0
    for i in range(rng): # Loop over CIS states
        if cis_energies[i]-cis_energies[0] > cutoff_energy:
            break
        print("State %s %04.3f eV \n" % (i,cis_energies[i]-cis_energies[0])) #print("State %s %04.3f eV \n" % (i,energy-cis_energies[0]))
        print("Excitation    CI Coef    CI C*rt(2)")
        spin = 0 # initialise total spin
        for j in range (cis_coeffs.shape[0]): # Loop over configurations in each CIS state   
        # if configuration is the ground determinant
            if j == 0: 
                if np.absolute(cis_coeffs[j,i]) > 1e-2:
                    print('|0>           %10.5f'  %(cis_coeffs[j,i]))
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
                continue
        # if configuration is |ibar->0bar>     
            elif j>0 and j<=ndocc:
                iorb = j-1
                str1 = str(ndocc-iorb) + "bar" #str(iorb) + "bar" #
                str2 = "0bar" #"3bar"#
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
       # if configuration is |0->j'> 
            elif j>ndocc and j<=ndocc+nunocc:
                jorb = j 
                str1 = "0" 
                str2 = str(jorb-ndocc)+"'" #str(jorb) #
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
        # if configuration is |Qi->j'>
            elif j>ndocc+nunocc and j<=ndocc+nunocc + ndocc*nunocc:
                iorb = int(np.floor((j-ndocc-nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-1)-iorb*nunocc+ndocc +1
                str1 = "Q " + str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'" 
                spin += 3.75*cis_coeffs[j,i]**2 # (S=1.5) 
         # if configuration is |D(S)i->j'> (bright doublet state)
            elif j>ndocc+nunocc + ndocc*nunocc and j<=ndocc+nunocc + 2*ndocc*nunocc:
                iorb = int(np.floor((j-ndocc-nunocc-ndocc*nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                str1 = "D(S) " +str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'"
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
         #if configuration is |D(T)i->j'> (dark doublet state)
            elif j>ndocc+nunocc + 2*ndocc*nunocc:
                iorb = int(np.floor((j-ndocc-nunocc-2*ndocc*nunocc-1)/nunocc))
                jorb = (j-ndocc-nunocc-2*ndocc*nunocc-1)-iorb*nunocc+ndocc +1
                str1 = "D(T) "+str(ndocc-iorb)
                str2 = str(jorb-ndocc)+"'"
                spin += 0.75*cis_coeffs[j,i]**2 # (S=0.5)
            if np.absolute(cis_coeffs[j,i]) > 1e-1:
                print("%s->%s %10.5f %10.5f " \
                %(str1,str2,cis_coeffs[j,i],cis_coeffs[j,i]*rt))
        if i==0:
            print("\n<S**2>: %04.3f" %spin)
            print("--------------------------------------------------------------------\n")
            continue
        osc = 2.0/3.0*((cis_energies[i]-cis_energies[0])/toev)*(mu0u[i,0]**2+mu0u[i,1]**2+mu0u[i,2]**2) 
        print("")
        print("TDMX:%04.3f   TDMY:%04.3f   TDMZ:%04.3f   Oscillator Strength:%04.3f   <S**2>: %04.3f" % (mu0u[i,0], mu0u[i,1], mu0u[i,2], osc, spin))
        print("--------------------------------------------------------------------\n")
        strng = strng + broaden(FWHM,osc,cis_energies[i]-cis_energies[0])
        #eqn+=broaden_as_fn(FWHM,osc,cis_energies[i]-cis_energies[0])
        osc_array[i]=osc
        s2_array[i]=spin
    strng = strng[1:]   
    return strng, cis_energies-cis_energies[0],osc_array,s2_array
