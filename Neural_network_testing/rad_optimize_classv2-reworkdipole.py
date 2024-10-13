import numpy as np
import scipy.optimize as optimize
import scipy.sparse.linalg as sp
import scipy.linalg as linalg
from datetime import datetime
from subprocess import getoutput
import sys
import os
import argparse
from rad_settings_opt import *

import time


parser = argparse.ArgumentParser()
parser.add_argument('weights_file', type = str, help = 'file containing weights and parameters which will be imported')
args = parser.parse_args()
weights_file = __import__(args.weights_file)
params = weights_file.params
weights = weights_file.weights
bds=weights_file.bds
jobname=args.weights_file
normalise_weights=weights_file.normalise_weights
hydrocbns=weights_file.hydrocbns
heterocyls=weights_file.heterocyls
hydcbn_data=weights_file.hydcbn_data
hetero_data=weights_file.hetero_data
opt=weights_file.opt


#def get_angles(file):
   # theta={'allyl':[0],'benzyl':[0],'dpm':[19,19],'trityl':[34.3,34.3,34.3],'dpxm':[35.9,35.9,29.7,34.2],'pdxm':[36.6,32.1,32.1,34.9,34.9],\
   #         'txm':[33.2,33.2,33.2,35,35,35], 'ttm':[48.9,48.9,48.9], 'ttm_1cz':[0,47.8,49.1,49.1,0,0,50.5],'pybtm':[47.47,49.23,49.20],'czbtm':[0,46.59,46.58,0,0,43.16],'ttm_pcz':[35.41,49.42,49.42,48.29,0,54.00],'ttm_1cz_anth':[63.7,0,0,0,0,0,49.49,49.38,49.31,52.19],'ttm_1cz_phanth':[40.54,0,49.52,49.3, 49.41,63.33,0,0,0,0,52.33],'ptm':[49.76 , 49.85, 49.64],'ttm_cz_cl2':[46.72,48.19,49.45,0,54.66,0,0]}[file]
  #  return theta

#Function to read geometries, stripping away any non-carbon atoms
#returns nx3 array of xyz coordinates
def read_geom(file):
    print("--------------------------------")
    print("Cartesian Coordinates / Angstrom")
    print("--------------------------------\n")
    f=open(file,'r')
    array=[] # array of coordinates of all atoms excluding hydrogen to be used as the atomic coordinates array for electronic structure calculation. If only carbon present it is array of carbon atoms
    array_n=[] # array of coordinates of specifically nitrogen atoms
    array_h=[] # array of coordinates of specifically hydrogen atoms
    array_cl=[] # array of coordinates of specifically chlorine atoms
    array_all=[] # array of coordinates of all atoms including hydrogen in order to calculate number of bonds to nitrogen. 
    #^ array_all has no functionality unless nitrogen present.
    atoms_c=[] # array of atomic symbols and atomic numbers of all atoms except hydrogen
    atoms_n=[]
    atoms_cl=[]
    atoms_h=[]
    natoms_c=0 #no of carbon atoms
    natoms_n=0 #no of nitrogen atoms
    natoms_cl=0 #no of chlorine atoms
    natoms_h=0 #no of hydrogen atoms
    for line in f:
        splt_ln=line.split()
        if line == '\n':
            break
        if splt_ln[0] in ["C","c"]:
            array.append(splt_ln[1:])
            atoms_c.append(['C', 12])
            print(line.rstrip('\n'))
            natoms_c += 1
        if splt_ln[0] in ["N","n"]:
            array_n.append(splt_ln[1:])
            atoms_n.append(['N', 14])
            print(line.rstrip('\n'))
            natoms_n += 1
        if splt_ln[0] in ["H","h"]:
            array_h.append(splt_ln[1:])
            atoms_h.append(['H', 1])
            natoms_h +=1
            print(line.rstrip('\n'))
        if splt_ln[0] in ["Cl","cl"]:
            array_cl.append(splt_ln[1:])
            atoms_cl.append(['Cl', 35.5])
            print(line.rstrip('\n'))
            natoms_cl += 1
    array = np.array(array)
    array = array.astype(np.float64)
    if natoms_n != 0:
        array_n = np.array(array_n)
        array_n = array_n.astype(np.float64)
        array = np.concatenate((array,array_n))
    if natoms_cl != 0:
        array_cl = np.array(array_cl)
        array_cl = array_cl.astype(np.float64)
        array = np.concatenate((array,array_cl))
    array_h = np.array(array_h)
    array_h = array_h.astype(np.float64)
    if natoms_h==0:
        array_all = array
    else:
        array_all=np.concatenate((array,array_h))
    atoms=atoms_c+atoms_n+atoms_cl+atoms_h # carbon then nitrogen then chlorine then hydrogen
    return array,atoms,array_all,natoms_c,natoms_n,natoms_cl,natoms_c+natoms_n+natoms_cl

#Function to calculate and return interatomic distances
# def distance(array):
#     dist_array=np.zeros((array.shape[0],array.shape[0]))
#     for i in range (np.shape(array)[0]):
#         for j in range(i+1,np.shape(array)[0]):
#             distance=0
#             for k in range (3):
#                 distance=distance+(array[i,k]-array[j,k])**2
#             distance= np.sqrt(distance)
#             dist_array[i,j]=distance
#             dist_array[j,i]=dist_array[i,j]
#     return dist_array
def distance(array):
    n = array.shape[0]
    dist_array = np.zeros((n, n))
    
    # Get upper triangular indices
    upper_tri_indices = np.triu_indices(n, k=1)
    
    # Calculate the distances
    separations = np.linalg.norm(array[upper_tri_indices[0]] - array[upper_tri_indices[1]], axis=1)
    
    # Assign the distances to the upper triangular part
    dist_array[upper_tri_indices] = separations
    
    # Reflect the upper triangular part to the lower triangular part
    dist_array += dist_array.T
    
    return dist_array

def adjacency(dist_array,cutoff):
    natoms=dist_array.shape[0]
    adj_mat=np.zeros((natoms,natoms))
    bond_count=0
    bond_list=[]
    for i in range(natoms):
        for j in range(i+1,natoms):
            if dist_array[i,j] < cutoff:
                adj_mat[i,j]=1
                adj_mat[j,i]=adj_mat[i,j]
                bond_count+=1
                bond_list.append([i,j])
    return adj_mat, bond_list

def array_intersect(lst1,lst2):
    lst3=[]
    for value in lst1:
        if value in lst2:
            lst3.append(value)
    return lst3

def dihedrals(natoms,atoms,coords,dist_array):
    a2,bond_list=adjacency(dist_array,cutoff)
    a3=np.dot(a2,a2)
    a4=np.dot(a3,a2)
    lst=[]
    for i in range(natoms):
        for j in range(i+1,natoms):
            if a4[i,j]!=0 and a3[i,j]==0 and a2[i,j]==0:
                lst.append([i,j])
                lst.append([j,i])
    angles={}
    for dihedral in lst:
        for bond in bond_list:
            if a2[dihedral[0],bond[0]]==1 and a2[dihedral[1],bond[1]]==1:
                if dist_array[bond[0],bond[1]]>single_bond_cutoff and atoms[bond[0]][0] in ['C','c'] and atoms[bond[1]][0] in ['C','c']:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta)
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
                elif dist_array[bond[0],bond[1]]>single_bond_cutoff_cn and array_intersect([atoms[bond[0]][0],atoms[bond[1]][0]],['N','n','N2','n2']) in [['N'],['n'],['N2'],['n2']]:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta)
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
    for bond in angles:
        avg_angle=sum(angles[bond])/len(angles[bond])
        angles.update({bond:avg_angle})
    return angles 
    
def compute_angle(dihedral,coords):
    # angle k-i-j-l
    rij = coords[dihedral[2],:]-coords[dihedral[1],:]
    rik = coords[dihedral[0],:]-coords[dihedral[1],:]
    rjl = coords[dihedral[3],:]-coords[dihedral[2],:]
    r1 = np.cross(rij,rik)
    r2 = np.cross(rij,rjl)
    # r1.r2 = |r1||r2|cost
    theta = np.arccos(np.dot(r1,r2)/(linalg.norm(r1)*linalg.norm(r2))) * 180/np.pi
     # return angle between 0 and (+)90 deg
    if theta > 90:
        theta = 180 - theta
    if theta < 0:
        theta = -theta
    return theta

def re_center(coords,atoms,coords_h): 
    com = np.zeros(3)
    summass=0
    for i in range(coords_h.shape[0]):
        for x in range(3):
            com[x] += atoms[i][1]*coords_h[i,x]
        summass+=atoms[i][1]
    for x in range(3):
        com[x] = com[x]/summass
    # heavy atoms only
    for i in range(coords.shape[0]):
        coords[i,:] -= com
    # all atoms (for subsequent perm dipole calculation)
    #for i in range(coords_h.shape[0]):
    #    coords_h[i,:] -= com
    return com,coords # return recentred coords for heavy atoms only

def ntype(array_all,atoms,natomsc,natomsn):
    nlist=[]
    for natom in range(natomsn):
        nbonds=-1 #will count distance between atom and itself as a bond so subtract 1 bond to account for this
        for iatom in range(array_all.shape[0]):
            distn=0
            for k in range(3):
                distn+=(array_all[natom+natomsc,k]-array_all[iatom,k])**2
            distn=np.sqrt(distn)
            if distn < cutoff:
                nbonds+=1
        nlist.append(nbonds-2)
        atoms[natom+natomsc][0]='N'+str(nbonds-1)
    return nlist,atoms   
   
#Function to group atoms into starred and unstarred
def conec(ncarb,array):
    star = []
    unst = []
    star.append(0)
    satom = [0]
    for n in range(ncarb):
        if len(star)+len(unst) == ncarb:
            break
        uatom = []
        for i in satom:
            for j in range(i+1,ncarb):
                if array[i,j] < cutoff and j not in unst:
                    uatom.append(j)
                    unst.append(j)
        satom = []            
        for i in uatom:
            for j in range(i+1,ncarb):
                if array[i,j] < cutoff and j not in star:
                    satom.append(j)
                    star.append(j)
    if len(star) < len(unst):
        print('Swapping starred and unstarred atoms ...')
        array = star
        star = unst
        unst = array
    print(' ')               
    print('Starred atoms: ' +str(star))
    print('Un-starred atoms: ' +str(unst)+'\n')
    return star, unst

# Routine to group bonding and antibonding orbitals into coulson-rushbrooke pairs
def order_orbs(ncarb,orbs,orb_energies,alt):
    print(' ')
    nbond = int((ncarb-1)/2)
    anti_list = list(range(nbond+1,ncarb))
    anti_list.reverse()
    pairs_list = []           
    search = False      
    for ibond in range(nbond):
        if abs(orb_energies[ibond+1] - orb_energies[ibond]) < 1e-6:
            print("degenerate orbitals %d and %d!"%(ibond+1,ibond+2))
            search=True
        elif ibond > 0:
            if abs(orb_energies[ibond-1] - orb_energies[ibond]) < 1e-6:
                print("degenerate orbitals %d and %d!"%(ibond+1,ibond))
                search=True
        if search == False:
            ianti = ncarb-ibond-1
            pairs_list.append([ibond,ianti])
            anti_list.remove(ianti)
            print('Coulson-Rushbrooke pair orbs %d, %d\n'%(ibond+1,ianti+1))
        if search == True:
            print('Searching for correct antibonding pair for orb %d ...'%(ibond+1))
            for ianti in anti_list: # guess orbital pair
                print("Trying antibonding orbital", ianti+1)
                if abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond])) < 1e-6: #if energies match
                    print("Absolute energies %4f eV and %4f eV match, difference = %4f eV"%(orb_energies[ibond],orb_energies[ianti],abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond]))))
                    pairs = tuple(zip(orbs[:,ibond],orbs[:,ianti])) # pairs of coeffs in bonding and antibonding orbital pair
                    for n,(icoeff,jcoeff) in enumerate(pairs): #compare coeffs
                        if abs(abs(icoeff) - abs(jcoeff)) > 1e-4: # if coeffs are not equal in magnitude start n loop again
                            print("Magnitude of coeffs not equal",abs(abs(icoeff) - abs(jcoeff)))
                            print('Searching for correct antibonding pair for orb %d ...'%(ibond+1))
                            break
                        if n == ncarb-1: # if all coefficients of two orbitals match in magnitude
                            pairs_list.append([ibond,ianti])
                            anti_list.remove(ianti)
                            print("Magnitudes of all orbital coefficients are within 1e-4")
                            print('Coulson-Rushbrooke pair orbs %d, %d\n'%(ibond+1,ianti+1))
                            search = False
                    if search == False:
                        break
                else:
                    print("absolute energies %4f eV and %4f eV do not match, difference = %4f eV"%(orb_energies[ibond],orb_energies[ianti],abs(abs(orb_energies[ianti]) - abs(orb_energies[ibond]))))
                if ianti==anti_list[len(anti_list)-1] and search==True: # if all antibonding orbitals are tried and none match the bonding orbital, warn user and switch off alternacy
                    print("\nWARNING!!: Could not find Coulson-Rushbrooke pair for orbital %d, switching off alternacy. If molecule is alternant, try lowering orbital coefficient matching threshold and re-run calculation. But examine the orbitals first!"%(ibond+1))
                    alt=False
                    return pairs_list, alt
    return pairs_list, alt                   
  
        #print("antibonding orbital",ianti+1)
        #pairs = tuple(zip(orbs[:,ibond],orbs[:,ianti])) # pairs of coeffs in bonding and antibonding orbital pair
# =============================================================================
#         for n,(icoeff,jcoeff) in enumerate(pairs): #compare coeffs
#             if abs(icoeff) - abs(jcoeff) > 1e-6: # if coeffs are not equal in magnitude
#                 print('Searching for correct antibonding pair for orb '+str(ibond+1))
#                 search = True
#                 for janti in range(nbond+1,ncarb): #loop over all antibonding orbs
#                     pairs2 = tuple(zip(orbs[:,ibond],orbs[:,janti]))
#                     for n,(icoeff,jcoeff) in enumerate(pairs2):
#                         if abs(icoeff) - abs(jcoeff) > 1e-6:
#                             break #break inner n loop and try another orbital
#                         elif n==ncarb-1:
#                             ant_list.append(janti)
#                             search = False
#                     if search == False:
#                         break #break janti loop
#                 if search == False:
#                             break #break outer n loop
#             elif n==ncarb-1:
#                 ant_list.append(ianti)
#                 print('Coulson-Rushbrooke pair orbs '+str(ibond+1)+','+str(ianti+1))
#     lst = tuple(zip(bdg_list,ant_list))
# =============================================================================


#Function to flip signs of orbital coeffs such that for every pair of bonding-antibonding orbitals, 
#starred atoms retain their sign in antibonding orbital and unstarred atoms have opposite sign 
def orb_sign(orbs,orb_energies,nelec,dist_array,alt):
    if alt==True:
        print('\nGrouping orbitals according to alternacy symmetry...')
        ncarb = orbs.shape[0]
        somo_energy = orb_energies[int((nelec-1)/2)]
        for i in range(orb_energies.shape[0]):
            orb_energies[i] = orb_energies[i] - somo_energy
        orb_list,alt = order_orbs(ncarb,orbs,orb_energies,alt)
    if alt==True:
        star,unst = conec(ncarb,dist_array)
        print('\nInverting orbital phases according to alternacy symmetry...\n')
        for i,ip in orb_list:
            for satom in star:
                if np.sign(orbs[satom,i]) != np.sign(orbs[satom,ip]):
                    orbs[satom,ip] = -1*orbs[satom,ip]
                    print('flipping sign orb '+str(ip)+' starred atom '+str(satom))
            for uatom in unst:
                if np.sign(orbs[uatom,i]) == np.sign(orbs[uatom,ip]):
                    print('flipping sign orb '+str(ip)+' unstarred atom '+str(uatom))
                    orbs[uatom,ip] = -1*orbs[uatom,ip]
    if np.sign(orbs[0,0]) == -1: # if orbital 0 has all -ve coeffs, make all +ve and invert all coeffs on all other orbitals
        orbs = np.multiply(orbs,-1) # as per Tim's alteration
    return orbs

#Function to form and return off-diagonal hopping contribution; use cutoff to determine nearest neighbors
def t_term(dist_array,natoms_c,natoms_n,natoms,n_list,theta,params):
    t=params[0][0]
    #tp=params[0][1]
    tp=params[0][0]*2.2/2.4
    alphan=params[1][0]
    tcn=params[1][1]
    alphan2=params[2][0]
    tcn2=params[2][1]
    tpcn2=params[2][1]*2.2/2.4 # change for cn2 hopping ratio
    alphacl=params[3][0]
    tccl=params[3][1]
    print("\nCarbon 1e params: t = %f tp = %f"%(t,tp))
    print("\nPyrrole Nitrogen 1e params: alphan2 = %f tcn2 = %f tpcn2 = %f"%(alphan2,tcn2,tpcn2))
    print("\nPyridine Nitrogen 1e params: alphan = %f tcn = %f tpcn = %f"%(alphan,tcn,tpcn2))
    print("\nChlorine 1e params: alphacl = %f tccl = %f"%(alphacl,tccl))
    array=np.zeros_like(dist_array)
    # C-C hopping
    ntheta=0
    print(dist_array[0,1])
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            if dist_array[i,j]<cutoff:
                if dist_array[i,j]<triple_bond_cutoff:
                    print("Triple bond")
                    array[i,j]=tt
                    array[j,i]=array[i,j]
                else:
                    array[i,j]=t
                    array[j,i]=array[i,j]
            if dist_array[i,j]<cutoff and dist_array[i,j]>single_bond_cutoff:
                print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*tp
                array[j,i]=array[i,j]  
                ntheta+=1
    # N and Cl hopping 
    # C-N hopping
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
            if dist_array[i,j]<cutoff:
                if n_list[j-natoms_c]==0:
                    array[i,j] = tcn
                    print("C-N1 bond")
                elif n_list[j-natoms_c]==1:
                    array[i,j] = tcn2
                    print("C-N2 bond")
                array[j,i] = array[i,j]
            if dist_array[i,j]<cutoff and dist_array[i,j]>single_bond_cutoff_cn and n_list[j-natoms_c]==1:
                print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*tpcn2
                array[j,i]=array[i,j]  
                ntheta+=1
    # N-N hopping
    for i in range (natoms_c,natoms_c+natoms_n):
        for j in range (i+1,natoms_c+natoms_n):
            if dist_array[i,j]<cutoff:
                if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
                    array[i,j] = tnn
                    print("N1-N1 bond")
                elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
                    array[i,j] = tnn2
                    print("N1-N2 bond")
                elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                    array[i,j] = tn2n2
                    print("N2-N2 bond")
                array[j,i] = array[i,j]  
    # C-Cl hopping
    for i in range (natoms_c):
        for j in range (natoms_c+natoms_n,natoms):
            if dist_array[i,j]<ccl_cutoff:
                array[i,j]=tccl
                print('C-Cl bond')
                array[j,i] = array[i,j]
    # N alpha (diagonal) terms
    for i in range(natoms_c,natoms_c+natoms_n):
        if n_list[i-natoms_c]==0:
            array[i,i] += alphan
            print("N1 atom %d"%(i+1))
        elif n_list[i-natoms_c]==1:
            array[i,i] += alphan2
            print("N2 atom %d"%(i+1))
    # Cl alpha (diagonal) terms
    for i in range(natoms_c+natoms_n,natoms):
        array[i,i]+=alphacl
        print("Cl atom %d"%(i+1))
    return array

#Function to form and return two-body repulsion (short and longe range) contribution
def v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params):
    #U=params[0][2]
    #r0=params[0][3]
    U=params[0][1]
    r0=params[0][2]
    Unn=params[1][2]
    r0nn=params[1][3]
    #Un2n2=params[2][3]
    #r0n2n2=params[2][4]
    Un2n2=params[2][2]
    r0n2n2=params[2][3]
    Uclcl=params[3][2]
    r0clcl=params[3][3]
    Ucn=(Unn+U)/2
    Ucn2=(Un2n2+U)/2
    Uccl=(U+Uclcl)/2
    Uncl=(Unn+Uclcl)/2
    Un2cl=(Un2n2+Uclcl)/2
    Unn2=(Un2n2+Unn)/2
    r0cn=(r0nn+r0)/2
    r0cn2=(r0n2n2+r0)/2
    r0ccl=(r0+r0clcl)/2
    r0ncl=(r0nn+r0clcl)/2
    r0n2cl=(r0n2n2+r0clcl)/2
    r0nn2=(r0n2n2+r0nn)/2
    print("\nCarbon 2e params: U = %f r0 = %f"%(U,r0))
    print("\nNitrogen 2e params: Un2n2 = %f r0n2n2 = %f"%(Un2n2,r0n2n2))
    print("\nChlorine 2e params: Uclcl= %f r0clcl = %f"%(Uclcl,r0clcl))
    print("\nMixed 2e params: Ucn2 = %f Uccl = %f Un2cl = %f r0cn2 =%f r0ccl = %f r0n2cl = %f"%(Ucn2,Uccl,Un2cl,r0cn2,r0ccl,r0n2cl))
    array=np.zeros_like(dist_array)
    # C-C repulsion
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            array[i,j]=U/(1+dist_array[i,j]/r0)
            array[j,i]=array[i,j] 
        array[i,i]=U
    # C-N repulsion
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
             if n_list[j-natoms_c]==0:
                 array[i,j]=Ucn/(1+dist_array[i,j]/r0cn)
             elif n_list[j-natoms_c]==1:
                 array[i,j]=Ucn2/(1+dist_array[i,j]/r0cn2)
             array[j,i]=array[i,j] 
    # N-N repulsion
    for i in range (natoms_c,natoms_c+natoms_n):
        for j in range (i+1,natoms_c+natoms_n):
             if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
                 array[i,j]=Unn/(1+dist_array[i,j]/r0nn)
             elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
                 array[i,j]=Unn2/(1+dist_array[i,j]/r0nn2)
             elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
                 array[i,j]=Un2n2/(1+dist_array[i,j]/r0n2n2)
             array[j,i]=array[i,j]
        # diagonal terms
        if n_list[i-natoms_c]==0:
            array[i,i]=Unn
        elif n_list[i-natoms_c]==1:
            array[i,i]=Un2n2
    # C-Cl repulsion
    for i in range(natoms_c):
        for j in range(natoms_c+natoms_n,natoms):
            array[i,j]=Uccl/(1+dist_array[i,j]/r0ccl)
            array[j,i]=array[i,j]
    # N-Cl repulsion
    for i in range(natoms_c,natoms_c+natoms_n):
        for j in range(natoms_c+natoms_n,natoms):
            if n_list[i-natoms_c]==0:
                array[i,j]=Uncl/(1+dist_array[i,j]/r0ncl)
            elif n_list[i-natoms_c]==1:
                array[i,j]=Un2cl/(1+dist_array[i,j]/r0n2cl)
            array[j,i]=array[i,j]
    # Cl-Cl repulsion
    for i in range(natoms_c+natoms_n,natoms):
        for j in range(i+1,natoms):
            array[i,j]=Uclcl/(1+dist_array[i,j]/r0clcl)
            array[j,i]=array[i,j]
        array[i,i]=Uclcl
    return array

#Function to form and return density matrix (for doublet monoradical)
def density(orbs,natoms,ndocc):
    density=2*np.dot(orbs[:,:ndocc], orbs[:,:ndocc].T) #doubly occ orbs  
    #optimise this with einsum:
    for u in range(natoms):
        for v in range(natoms):
            density[u,v] += orbs[u,ndocc]*orbs[v,ndocc] #somo 
    return density

#Function to form and return open-shell Fock matrix
def fock(repulsion,hopping,density,natoms_c,natoms_n,natoms,nlist):
    fock_mat=np.zeros_like(repulsion)
    for i in range (natoms):
        for j in range (i,natoms):
            if i==j:
                mylist=[]
                for k in range (natoms): 
                    mylist.append(k)
                mylist.remove(i)
                for n in mylist:
                    if n>=natoms_c and n<natoms_c+natoms_n: #N atom
                        zk = nlist[n-natoms_c]+1
                    elif n>=natoms_c+natoms_n: # Cl atom
                        zk=2
                    else: # Carbon
                        zk=1
                    fock_mat[i,j] += (density[n,n]-zk)*repulsion[i,n]
                fock_mat[i,j] += 0.5*density[i,j]*repulsion[i,j]
            else:
                fock_mat[i,j]=-0.5*density[i,j]*repulsion[i,j]
                fock_mat[j,i]=fock_mat[i,j]
    fock_mat=fock_mat+hopping
    return fock_mat

def compute_j00(orbs,repulsion,ndocc):
    J00 = 0
    for l in range(orbs.shape[0]):
        for m in range(orbs.shape[0]):
            J00 += orbs[l,ndocc]**2 * orbs[m,ndocc]**2 * repulsion[l,m]
    #print(J00)
    return J00

#Function to calculate open-shell SCF energy
def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    J00 = compute_j00(orbs,repulsion,ndocc)
    return 0.5*(np.dot(density.flatten(),hopping.flatten())+np.dot(density.flatten(),fock_mat.flatten())) - 0.25*J00

#Main HF function
def main_scf(file,params):
    print("                    ---------------------------------")
    print("                    | Radical ExROPPP Calculation |")
    print("                    ---------------------------------\n")
    print("Molecule: "+str(file)+" radical\n")
#read in geometry and form distance matrix
    coord,atoms_array,coord_w_h,natoms_c,natoms_n,natoms_cl,natoms=read_geom(file)
    dist_array=distance(coord)
    n_list,atoms=ntype(coord_w_h,atoms_array,natoms_c,natoms_n)
    nelec=natoms + sum(n_list) + natoms_cl #each pyrolle type N contributes 1 additional e-, so does Cl
    ndocc = int((nelec-1)/2) # no. of doubly-occupied orbitals
    print("\nThere are %d heavy atoms."%natoms)
    print("There are %d electrons in %d orbitals.\n"%(nelec,natoms))
#compute array of dihedral angles for given molecule (originaly used predefined dictionary of angles but now they are computed directly)
    angles=dihedrals(natoms_c+natoms_n+natoms_cl,atoms_array,coord,dist_array)
#call functions to get 1/2-body "integrals"
    hopping=t_term(dist_array,natoms_c,natoms_n,natoms,n_list,angles,params)
    repulsion=v_term(dist_array,natoms_c,natoms_n,natoms,n_list,params)
#Diagonalize Huckel Hamiltonian to form initial density guess
    guess_evals,evecs=np.linalg.eigh(hopping)
    guess_dens=density(evecs,natoms,ndocc)
#iterate until convergence 
    energy1=0
    print("\n-------------------------------------")
    print("Restricted Open-shell PPP Calculation")
    print("-------------------------------------\n")
    print("Starting SCF cycle...\n")
    print("Iter   Energy        Dens Change      Energy Change")
    print("-----------------------------------------------------")
    for iter in range (501):
        if iter==500:
            print("\nEnergy not converged after 500 cycles")
            break
        fock_mat=fock(repulsion,hopping,guess_dens,natoms_c,natoms_n,natoms,n_list)
        evals,orbs=np.linalg.eigh(fock_mat)
        dens=density(orbs,natoms,ndocc)
        energy2=energy(hopping,repulsion,fock_mat,dens,orbs,ndocc)
        conv_crit=np.absolute(guess_dens-dens).max()
        print(iter, energy2, conv_crit,energy2-energy1)
        cutoff=0.0000001
        if conv_crit<cutoff:
            return coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy2,hopping,repulsion,evals,orbs,fock_mat
        if energy2>energy1:
            print('\nEnergy rises!')
        energy1=energy2
        guess_dens=dens

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

# def absorb(x,energy_array,osc_array):
#     if brdn_typ == 'energy' and line_typ == 'lorentzian':
#         absorb=0
#         for i, energy in enumerate(energy_array):
#             absorb-=osc_array[i]*1/(1+(((evtonm/energy)-x)/(0.5*FWHM*(evtonm/energy)*x))**2)
#     return absorb


  
#def fabs(x,strng): 
 #   f = eval(strng)
#    return - f

def write_gnu(strng,file):
    f=open('gnuplot_script_%s_%s'%(jobname,file),'w')
    f.write("#simulated spectrum\n")
    f.write("set term pdf size 6,4\n")
    f.write("unset key\n")
    f.write("set output '%s.pdf'\n" %(file))
    f.write("set xrange [200:700]\n")
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',18'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',18'\n")
    f.write("set xtics font ',18'\n")
    f.write("set ytics font ',18'\n")
    f.write("set bmargin 4\n")
    f.write("p %s lw 3 dt 1" %strng)
    f.close()
    return
    
def read_gmc_data(molecule):
    filename = molecule + str('_data.txt')
    if os.path.isfile(filename) == False:
        print("There is no file '%s'"%filename)
        return ""
    energies=[]
    osc=[]
    f=open(filename,'r')
    for line in f:
        splt_ln=line.split()
        if line == '\n':
            break
        energies.append(float(splt_ln[0]))
        osc.append(float(splt_ln[1]))
    strng=""
    for i in range(len(energies)):
       strng += broaden(FWHM,osc[i],energies[i])
    strng = strng[1:]
    return strng

def multi_gnu(figure,rnge):
    figurename=""
    for molecule in figure:
        figurename += molecule +'_'
    figurename = figurename[:-1]
    f=open('gnuplot_script_'+figurename,'w')
    f.write("# ROPPP simulated spectrum\n")
    f.write("set output '%s.pdf'\n" %(figurename))
    f.write("set term pdf size 6,8\n")
    f.write("set multiplot\n")
    f.write("set key right top font ',16'\n")
    f.write("set xrange %s\n"%rnge)
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',16'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',16'\n")
    f.write("set xtics font ',14'\n")
    f.write("set ytics font ',14'\n")
    f.write("set size 0.9, 0.45\n")
    f.write("set lmargin 10\n")
    f.write("set rmargin 0\n")
    f.write("set tmargin 0\n")
    f.write("set bmargin 5\n")
    f.write("set title 'ROPPP'\n")
    f.write("p")
    f.close()
    return figurename

def append_gnu(figurename,molecule,strng):
    if len(strng) < 2:
        return
    print("Plotting simulated spectrum of the %s radical ..."%molecule)
    f=open('gnuplot_script_'+figurename,'a')
    f.write(" %s lw 2 dt 1 lc rgb '%s' title '%s'," %(strng,linecolours[molecule],molecule))
    #if method == 'roppp':
    #    print("Plotting ROPPP simulated spectrum of the %s radical ..."%molecule)
      #  f=open('gnuplot_script_'+figurename,'a')
       # f.write(" %s lw 2 dt 1 lc rgb '%s' title '%s - ROPPP'," %(strng,linecolour,molecule))  
    #elif method == 'gmc':
      #  print("Plotting GMC-QDPT simulated spectrum of the %s radical ..."%molecule)
       # f=open('gnuplot_script_'+figurename,'a')
       # f.write(" %s lw 2 dt 4 lc rgb '%s' title '%s - GMC-QDPT', " %(strng,linecolour,molecule))
    f.close()
    print("done!")
    return  
    
def plot_all(gnudata):
    # ROPPP FIGURE 1
    figurename = multi_gnu(figure1,'[150:450]')
    for molecule in figure1:
        append_gnu(figurename,molecule,gnudata[molecule][0]) 
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\n# GMC-QDPT simulated spectrum\n")
    f.write("set noxtics\n")
    f.write("set noxlabel\n")
    f.write("set key right top\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units'\n")
    f.write("set origin 0,0.43\n")
    f.write("set lmargin 10\n")
    f.write("set rmargin 0\n")
    f.write("set title 'GMC-QDPT'\n")
    f.write("p")
    f.close()
    # GMC-QDPT FIGURE 1
    for molecule in figure1:
        append_gnu(figurename,molecule,gnudata[molecule][1])
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\nunset multiplot\n")
    f.close()
    # ROPPP FIGURE 2
    figurename = multi_gnu(figure2,'[200:500]')
    for molecule in figure2:
        append_gnu(figurename,molecule,gnudata[molecule][0]) 
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\n# GMC-QDPT simulated spectrum\n")
    f.write("set noxtics\n")
    f.write("set noxlabel\n")
    f.write("set key right top\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units'\n")
    f.write("set origin 0,0.43\n")
    f.write("set lmargin 10\n")
    f.write("set rmargin 0\n")
    f.write("set title 'GMC-QDPT'\n")
    f.write("p")
    f.close()
    # GMC-QDPT FIGURE 2
    for molecule in figure2:
        append_gnu(figurename,molecule,gnudata[molecule][1])
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\nunset multiplot\n")
    f.close()
    # ROPPP FIGURE 3
    figurename = multi_gnu(figure3,'[300:500]')
    for molecule in figure3:
        append_gnu(figurename,molecule,gnudata[molecule][0]) 
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\n# GMC-QDPT simulated spectrum\n")
    f.write("set noxtics\n")
    f.write("set noxlabel\n")
    f.write("set key right top\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units'\n")
    f.write("set origin 0,0.43\n")
    f.write("set lmargin 10\n")
    f.write("set rmargin 0\n")
    f.write("set title 'GMC-QDPT'\n")
    f.write("p")
    f.close()
    #GMC-QDPT FIGURE 3
    for molecule in figure3:
        append_gnu(figurename,molecule,gnudata[molecule][1])
    f=open('gnuplot_script_'+figurename,'a')
    f.seek(0, 2)              # seek to end of file; f.seek(0, os.SEEK_END) is legal
    f.seek(f.tell() - 2, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
    f.truncate()
    f.write("\nunset multiplot\n")
    f.close()
      
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
    # for i in range (ndocc): 
    #     cish[0,i+1] = 0.5*rep_tens[i,o0,o0,o0] 
    #     cish[i+1,0] = cish[0,i+1]
    cish[0, 1:ndocc+1] = 0.5 * rep_tens[:ndocc, o0, o0, o0]
    cish[1:ndocc+1, 0] = cish[0, 1:ndocc+1] 
    #3 <0|H|0->j'>
    # for j in range (nunocc):
    #     cish[0,j+ndocc+1] = -0.5*rep_tens[o0,j+ndocc+1,o0,o0] 
    #     cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    cish[0, ndocc+1:ndocc+1+nunocc] = -0.5 * rep_tens[o0, ndocc+1:ndocc+1+nunocc, o0, o0]
    cish[ndocc+1:ndocc+1+nunocc, 0] = cish[0, ndocc+1:ndocc+1+nunocc]

    
    #4 <0|H|Q i->j'> ALL ZERO  
    #5 <0|H|D(+)i->j'> ALL ZERO (only depends on Fij')
    #6 <0|H|D(-)i->j'>
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+2*ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = 3/(np.sqrt(6)) *rep_tens[i,o0,o0,j]
        # cish[nn,0] = cish[0,nn]
    cish[ndocc + nunocc + 2 * ndocc * nunocc + 1 : ndocc * nunocc + ndocc + nunocc + 2 * ndocc * nunocc + 1 , 0] = \
    cish[0, ndocc + nunocc + 2 * ndocc * nunocc + 1 : ndocc * nunocc + ndocc + nunocc + 2 * ndocc * nunocc + 1]
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

def rad_calc(file,params):
    #Call HF driver--Comment out last line to stop here
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,one_body,two_body,orb_energy,hf_orbs,fock_mat=main_scf(file,params)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    hf_orbs = orb_sign(hf_orbs,orb_energy,nelec,dist_array,alt)
    print("\n--------------------------")
    print("Converged ROPPP Orbitals")
    print("--------------------------\n")
    natoms=np.shape(coord)[0]
    for iorb in range(natoms):
        print('orbital number', iorb + 1, 'energy', orb_energy[iorb]-orb_energy[int((nelec-1)/2)])
        print(np.around(hf_orbs[:, iorb], decimals=2)) #print(np.around(guess_orbs[:, iorb], decimals=2))

            #########################################################
             # PRINTING OF MOLECULAR ORBITALS BASED ON GAMESS OUTPUT #
             #########################################################
    atomic_numbers=[]
    for atom in atoms_array:
        number={"C":6.0,"c":6.0,"H":1.0,"h":1.0,"N":7.0,"n":7.0,"N1":7.0,"n1":7.0,"N2":7.0,"n2":7.0,"Cl":17.0,"cl":17.0,"CL":17.0}[atom[0]]
        atomic_numbers.append([atom[0],number])
    f=open('%s.out'%file,'w')
    f.write("\n")
    f.write("\nGAMESS COORDINATES FORMAT")
    f.write("\n")
    f.write("\n ATOM      ATOMIC                      COORDINATES (BOHR)")
    f.write("\n           CHARGE         X                   Y                   Z")
    #for i,atom in enumerate(atoms_array):
    for i in range(natoms_c+natoms_n+natoms_cl):
        f.write("\n %s           %d     %f            %f            %f"%(atoms_array[i][0],atomic_numbers[i][1],coord[i,0]*tobohr,coord[i,1]*tobohr,coord[i,2]*tobohr))
    f.write("\n                      ")
    f.write("\n     ATOMIC BASIS SET")
    f.write("\n     ----------------")
    f.write("\n ")
    f.write("\n ")
    f.write("\n ")
    f.write("\n  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)")
    f.write("\n ")
    n1=1
    n2=1
    for i,atom in enumerate(atoms_array):
        if atom[0] == 'C':
            f.write("\n C         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s            27.3850330    0.430128498301"%(str(n1+i),str(n2+i)))
            f.write("\n     %2s   S     %3s             4.8745221    0.678913530502"%(str(n1+i),str(n2+i+1)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             1.1367482    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             0.2883094    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            n1+=1
            n2+=3
        if atom[0] == 'Cl':
            f.write("\n Cl         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s           229.9441039    0.430128498301"%(n1+i,n2+i))
            f.write("\n     %2s   S     %3s            40.9299346    0.678913530502"%(n1+i,n2+i+1))
            f.write("\n ")
            f.write("\n     %2s   L     %3s            15.0576101    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             3.8190075    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             0.8883464   -0.298398604487    0.348047191182"%(str(n1+i+2),str(n2+i+4)))
            f.write("\n     %2s   L     %3s             0.3047828    1.227982887359    0.722252322062"%(str(n1+i+2),str(n2+i+5)))
            n1+=2
            n2+=5
        f.write("\n ")  
    for imo in range(hf_orbs.shape[0]):
        f.write("\n ")
        f.write("\n          ------------")
        f.write("\n          EIGENVECTORS")
        f.write("\n          ------------")
        f.write("\n ")
        f.write("\n                      %s    "%str(imo+1))
        f.write("\n                   %4f "%(orb_energy[imo]-orb_energy[int((nelec-1)/2)]))
        f.write("\n                     A     ")# symmetry (A is default for c1)
        kao=1
        for jatom, atom in enumerate(atoms_array):
            if atom[0]=='C':
                if file=='allyl' or file=='benzyl':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    %6f"%(str(kao+3),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                elif file=='dpm' or file=='dpxm' or file=='pdxm':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                else:
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    kao+=5
            if atom[0]=='Cl':
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+1),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+5),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+6),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+7),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    %6f"%(str(kao+8),str(jatom+1),hf_orbs[jatom,imo]))
                kao+=9
        f.write("\n  ...... END OF ROHF CALCULATION ......")
    f.write("\n ")
    f.close()
    # check that fock matrix is diagonalized
    fock_mo = np.dot(hf_orbs.T,np.dot(fock_mat,hf_orbs))
    for i in range(fock_mo.shape[0]):
        for j in range(fock_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Fock matrix not converged!")
                print("\nFock Matrix:")
                print(fock_mo)
                sys.exit()
    # check the density matrix
    dens_mat=density(hf_orbs,natoms,ndocc)
    dens_mo = np.dot(hf_orbs.T,np.dot(dens_mat,hf_orbs))
    print('\nOrbital occupation numbers:')
    for i in range(dens_mo.shape[0]):
        print("%d: %f"%(i+1,dens_mo[i,i]))
        for j in range(dens_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Density matrix not converged!")
                print("\nDensity Matrix:")
                print(dens_mo)
                sys.exit()
    if sum(n_list)==0 and natoms_cl==0:
        #strng,ci_energies_array,osc_array = cis(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs,'cis')  
        #strng,ci_energies_array,osc_array = cis(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs,'cisd')  
        strng,ci_energies_array,osc_array,s2_array= cisd_rot(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs)
    else:
        strng,ci_energies_array,osc_array,s2_array = hetero_cisd_rot(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs)
    return strng,ci_energies_array,osc_array,s2_array  #return gnuplot data for plotting spectrum

def write_comp(d1_energies: dict, bright_energies: dict):
    exp_energies={'allyl': [3.07, 5.0], 'benzyl': [2.755548342, 3.999989529, 4.769218285], 'dpm': [2.370930696, 3.69046653], 'trityl': [2.405593703, 3.595990587], 'dpxm': [2.070794579, 3.310791333, 3.595990587], 'pdxm': [2.033594677, 3.02559208, 3.447190976], 'txm': [2.033594677, 2.951192275]}
    gmcqdpt_energies={'allyl': [3.4465528, 6.4685596], 'benzyl': [3.0465578,4.2007681, 5.2475845], 'dpm': [2.556545,4.1321081], 'trityl': [2.6301691, 3.9113602], 'dpxm': [2.3973264, 3.5094722, 3.9095689], 'pdxm': [2.3329004, 3.423308, 3.873217], 'txm': [2.303476, 3.3438274]}
    gmcqdpt_energies2={'allyl': [3.4465528, 6.4685596,6.06845868432759], 'benzyl': [3.0465578,4.2007681, 5.2475845,4.42133703101611], 'dpm': [2.556545,4.1321081,4.212428922], 'trityl': [2.6301691, 3.9113602,4.333618338287], 'dpxm': [2.3973264, 3.5094722, 3.9095689], 'pdxm': [2.3329004, 3.423308, 3.873217], 'txm': [2.303476, 3.3438274]}
    d1_file=open("compD1_%s.txt"%jobname,"w")
    bright_file=open("compBrght_%s.txt"%jobname,"w")
    for molecule, energies in d1_energies.items():
        d1_file.write(f'{molecule}\n')
        d1_file.write(f'{energies}\n')
        # for i, data in enumerate(hydcbn_data[molecule]):
        # for i in range(len(hydcbn_data[molecule])):
        #     f.write("\n%f\t%f"%(evtonm/hydcbn_data[molecule][i],calc_energies2[molecule][i]))
        # d1_file.write('\n')
    d1_file.close()

    for molecule, energies in bright_energies.items():
        bright_file.write(f'{molecule}\n')
        bright_file.write(f'{energies}\n')
        # for i, data in enumerate(hetero_data[molecule]):
        # bright_file.write('\n')
        # f.write(f'{evtonm/data}),{calc_energies2[molecule]}')
        # if molecule in ['ttm_cz_ph','ttm_bcz','ttm_cz_ph2','ttm_dbcz','ttm_id3','c_16']:
        #     f.write("\n%f\t%f"%(evtonm/hetero_data[molecule][0],calc_energies2[molecule][0]))
        # else:
        #     for i in range(2):
        #         f.write("\n%f\t%f"%(evtonm/hetero_data[molecule][i],calc_energies2[molecule][i]))
        
    bright_file.close()

    return



class ParameterOptimization:
    def __init__(self, params, weights, atom, jobname, normalise_weights, hydrocbns, heterocyls, hydcbn_data, hetero_data, opt, bounds):
        self.params = params
        self.weights = np.array(weights).astype(np.float64) / sum(weights)
        self.atom = atom
        self.jobname = jobname
        self.normalise_weights = normalise_weights
        self.hydrocbns = hydrocbns
        self.heterocyls = heterocyls
        self.hydcbn_data = hydcbn_data
        self.hetero_data = hetero_data
        self.opt = opt
        # self.total_fit = 0
        self.Bright_energy = {}
        self.D1_energy = {}  # this is for heterocycles
        self.bds = bounds
        self.fit_iter = 0
        
        self.iteration = 0

    def absorb(self, x,strng):
        absorb=eval(strng)
        return np.float64(-1*absorb) 
    
    def minimize_absorb(self, wlnm, string):
        result = optimize.minimize(self.absorb, wlnm, args=(string,))
        abs_val = -1 * np.float64(result['fun'])
        return abs_val, np.float64(result['x'])

    def log_parameters(self, message):
        with open(f"{self.jobname}.txt", "a") as f:
            f.write(message)

    def print_parameters(self, type: str, active_params):
        

        C_str = (
            f"\nCarbon parameters in current iteration: \n"
            f"t = {active_params[0][0]:.6f}, tp = {active_params[0][0] * 2.2 / 2.4:.6f}, "
            f"U = {active_params[0][1]:.6f}, r0 = {active_params[0][2]:.6f}\n"
            f'sanity = {active_params[0][3]}\n'
        )
        N_str = (
            f"\nPyridine Nitrogen parameters in current iteration: \n"
            f"alphan = {active_params[1][0]:.6f}, tcn = {active_params[1][1]:.6f}, "
            f"Unn = {active_params[1][2]:.6f}, r0nn = {active_params[1][3]:.6f}\n"
            f"\nPyrole Nitrogen parameters in current iteration: \n"
            f"alphan2 = {active_params[2][0]:.6f}, tcn2 = {active_params[2][1]:.6f}, "
            f"tpcn2 = {active_params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {active_params[2][2]:.6f}, "
            f"r0n2n2 = {active_params[2][3]:.6f}\n"
            f"\nChlorine parameters in current iteration: \n"
            f"alpha_cl = {active_params[3][0]:.6f}, tccl = {active_params[3][1]:.6f}, "
            f"Uclcl = {active_params[3][2]:.6f}, r0clcl = {active_params[3][3]:.6f}\n"
        )

        gen_str = f"\nFitting data: {self.hydcbn_data} \n {self.hetero_data}"
        if type == 'initialization':
            message = C_str + N_str + gen_str
            print(message)
            self.log_parameters(message)
        # message = (
        #     f"\nCarbon parameters in current iteration: \n"
        #     f"t = {self.params[0][0]:.6f}, tp = {self.params[0][0] * 2.2 / 2.4:.6f}, U = {self.params[0][1]:.6f}, r0 = {self.params[0][2]:.6f}\n"
        #     f"\nPyridine Nitrogen parameters in current iteration: \n"
        #     f"alphan = {self.params[1][0]:.6f}, tcn = {self.params[1][1]:.6f}, Unn = {self.params[1][2]:.6f}, r0nn = {self.params[1][3]:.6f}\n"
        #     f"\nPyrole Nitrogen parameters in current iteration: \n"
        #     f"alphan2 = {self.params[2][0]:.6f}, tcn2 = {self.params[2][1]:.6f}, tpcn2 = {self.params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {self.params[2][2]:.6f}, r0n2n2 = {self.params[2][3]:.6f}\n"
        #     f"\nChlorine parameters in current iteration: \n"
        #     f"alpha_cl = {self.params[3][0]:.6f}, tccl = {self.params[3][1]:.6f}, Uclcl = {self.params[3][2]:.6f}, r0clcl = {self.params[3][3]:.6f}\n"
        #     f"\nFitting data: {self.hydcbn_data, self.hetero_data}"
        # )
        elif type == 'hydrocbns':
            message = C_str
            print(message)
            self.log_parameters(message)

        elif type == 'hetereo':
            message = N_str
            print(message)
            self.log_parameters(message)

    def fitness(self, active_params, inactive_params, weights, atom):
        hydrocbns_w_gmc = ['benzyl', 'allyl', 'dpm', 'trityl']
        hydrocbns_w_2bright = ['benzyl', 'dpxm', 'pdxm']
        hetereo_only_d1 = ['ttm_cz_ph', 'ttm_bcz', 'ttm_cz_ph2', 'ttm_dbcz', 'ttm_id3', 'c_16']
        self.fit_iter = 0
        total_fit = 0
        # print(time.perf_counter()-start_time)
        # print(active_params)
        time_now_iter = time.perf_counter()
        self.print_parameters(type='initialization', active_params = np.array(active_params).reshape(4,-1))

        # Carbon routine
        for file in self.hydrocbns:
            print(f'{file} is being optimized')
            # self.print_parameters(type='hydrocbns')
            strng, energies, oscs, s2_array = rad_calc(file, np.array(active_params).reshape(4,-1))
            if len(self.hydcbn_data[file]) == 1:  # when only D1 data is available

                self.process_hydrocarbons_only_d1(file, energies, oscs, s2_array, total_fit, weights, strng)
    
            # Process hydrocarbons with GMC and one bright state
            elif file in hydrocbns_w_gmc and file not in hydrocbns_w_2bright:
                self.process_hydrocarbons_with_gmc(file, energies, oscs, s2_array, total_fit, weights, strng)

            # Process with 2 bright states and has GMC
            elif file in hydrocbns_w_gmc and file in hydrocbns_w_2bright:
                self.process_hydrocarbons_with_two_bright_states(file, energies, oscs, s2_array, total_fit, weights, strng)

            # Process with two bright state but no GMC
            elif file in hydrocbns_w_2bright and file not in hydrocbns_w_gmc:
                self.process_hydrocarbons_with_two_bright_states_no_gmc(file, energies, oscs, total_fit, weights, strng)

            # Process with bright and d1
            elif len(self.hydcbn_data[file]) == 2:
                self.process_hydrocarbons(file, energies, oscs, total_fit, weights, strng)

        # Heterocycle routine
        for file in self.heterocyls:
            print(f'{file} is being optimized')
            # self.print_parameters(type='hetereo')
            strng, energies, oscs, s2_array = rad_calc(file, np.array(active_params).reshape(4,-1))
            
            if file in hetereo_only_d1:
                self.process_heterocycles_with_only_d1(file, energies, total_fit, weights, strng)
            elif len(self.hetero_data[file]) == 2:
                self.process_heterocycles_with_two_states(file, energies, oscs, total_fit, weights, strng)
            elif len(self.hetero_data[file]) == 3:
                self.process_heterocycles_with_three_data(file, energies, oscs, total_fit, weights, strng)
            else:
                self.process_heterocycles_full(file, energies, oscs, s2_array, total_fit, weights, strng)
                
        write_comp(d1_energies = self.D1_energy,bright_energies=self.Bright_energy)
        time_needed = time.perf_counter() - time_now_iter
        self.iteration += 1
        self.log_parameters(f'\niteration {self.iteration} need {time_needed} seconds\n')
        if self.opt == False:
            sys.exit()
        return total_fit

    def optimize_params(self):
        print("---------------------- Parameter Fitting  ----------------------")
        if self.atom in ['cl', 'CL', 'Cl']:
            self.optimize_chlorine()
        elif self.atom in ['c', 'C']:
            self.optimize_carbon()
        elif self.atom in ['n2', 'N2']:
            self.optimize_pyrrole()
        elif self.atom in ['n', 'N']:
            self.optimize_pyridine()
        elif self.atom == 'all':
            self.optimize_all()

    def log_and_print(self, file, energies, d1_exp=None, d1_calc=None, bright_exp=None, bright_calc=None, bright_exp2=None, bright_calc2=None, q1_exp=None, q1_calc=None, fratio_exp=None, fratio_calc=None, fitness=None, total_fit=None, testing = None):
        messages = []
        
        if d1_exp is not None:
            messages.append(f'd1 state exp / nm = {evtonm / d1_exp:.6f}')
        if d1_calc is not None:
            messages.append(f'd1 state calc / nm = {evtonm / d1_calc:.6f}')
        
        if bright_exp is not None:
            messages.append(f'bright state exp / nm = {evtonm / bright_exp:.6f}')
        if bright_calc is not None:
            messages.append(f'bright state calc / nm = {evtonm / bright_calc:.6f}')
        
        if bright_exp2 is not None:
            messages.append(f'bright state exp / nm = {evtonm / bright_exp2:.6f}')
        if bright_calc2 is not None:
            messages.append(f'bright state calc / nm = {evtonm / bright_calc2:.6f}')
        
        if q1_exp is not None:
            messages.append(f'q1 gmc-qdpt / nm = {evtonm / q1_exp:.6f}')
        if q1_calc is not None:
            messages.append(f'q1 state = {evtonm / q1_calc:.6f}')
        
        if fratio_exp is not None:
            messages.append(f'intensity ratio = {fratio_exp:.6f}')
        if fratio_calc is not None:
            messages.append(f'fosc ratio = {fratio_calc:.6f}')
        
        if fitness is not None:
            messages.append(f'fitness = {fitness:.6f}')
        if total_fit is not None:
            messages.append(f'total fitness = {total_fit:.6f}')
        if testing is not None:
            messages.append(f'{testing}')

        message = "\n".join(messages)
        print(f'\n{file}\n{message}')
        self.log_parameters(f"\n{file}\n{message}\n")

    def process_heterocycles_full(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        q1_gmcqdpt = evtonm / self.hetero_data[file][2]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 100
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]
        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp,bright_calc= bright_calc, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp, bright_calc]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})
    def process_hydrocarbons_only_d1(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hydcbn_data[file][0]

        fitness = weights[0] * (d1_calc - d1_exp) ** 2 if self.normalise_weights == 'per_state' else 0
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_hydrocarbons_with_gmc(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp = evtonm / self.hydcbn_data[file][1]
        q1_gmcqdpt = evtonm / self.hydcbn_data[file][2]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 100
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]
        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp,bright_calc= bright_calc, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp, bright_calc]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_hydrocarbons_with_two_bright_states(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp1 = evtonm / self.hydcbn_data[file][1]
        bright_exp2 = evtonm / self.hydcbn_data[file][2]
        q1_gmcqdpt = evtonm / self.hydcbn_data[file][3]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs

        imax1 = np.argmax(oscs)
        oscs[imax1] = 0
        imax2 = np.argmax(oscs)
        bright_calc1 = energies[min(imax1, imax2)]
        bright_calc2 = energies[max(imax1, imax2)]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp1, bright_calc=bright_calc1, bright_exp2=bright_exp2, bright_calc2=bright_calc2, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness, total_fit=self.fit_iter)
        

        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp1, bright_calc1, bright_exp2, bright_calc2]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_hydrocarbons_with_two_bright_states_no_gmc(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp1 = evtonm / self.hydcbn_data[file][1]
        bright_exp2 = evtonm / self.hydcbn_data[file][2]

        d1_calc = energies[1]

        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs

        imax1 = np.argmax(oscs)
        oscs[imax1] = 0
        imax2 = np.argmax(oscs)
        bright_calc1 = energies[min(imax1, imax2)]
        bright_calc2 = energies[max(imax1, imax2)]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2  
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp1, bright_calc=bright_calc1, bright_exp2=bright_exp2, bright_calc2=bright_calc2, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp1, bright_calc1, bright_exp2, bright_calc2]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_hydrocarbons(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp = evtonm / self.hydcbn_data[file][1]

        d1_calc = energies[1]
        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]

        if self.normalise_weights == 'per_state' :
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp, bright_calc]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_heterocycles_with_only_d1(self, file, energies, total_fit, weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hetero_data[file][0]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2  
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, fitness=fitness, total_fit=self.fit_iter)

        write_gnu(strng, file)
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_heterocycles_with_two_states(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        d1_calc = energies[1]

        # other_brt_states = [self.minimize_absorb(wlnm, strng) for wlnm in [375, 500, 700]]
        
        # abs_max, lmax = max(other_brt_states, key=lambda x: x[1])
        # other_brt_states = [state for state in other_brt_states if state != (lmax, abs_max)]

        # bright_calc = np.float64(evtonm / lmax)
        other_brt_states=[]
        abs_max=0
        for wlnm in [375,500,700]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        other_brt_states.remove([lmax,abs_max])
        bright_calc=np.float64(evtonm/lmax)

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fitness=fitness, total_fit=self.fit_iter)
        
        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp, bright_calc]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})

    def process_heterocycles_with_three_data(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        fratio_exp = self.hetero_data[file][2]

        d1_calc = energies[3] if file in ['ttm_1cz_anth', 'ttm_1cz_phanth'] else energies[1]

        threshold_absorb = evtonm / 300
        index = np.argmax(np.array(energies) > threshold_absorb)
        if energies[index] > threshold_absorb:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]

        # other_brt_states = [self.minimize_absorb(wlnm, strng) for wlnm in [375, 500, 700]]
        # abs_max, lmax = max(other_brt_states, key=lambda x: x[1])
        # other_brt_states = [state for state in other_brt_states if state != (lmax, abs_max)]

        # bright_calc = np.float64(evtonm / lmax)
        # fratio_calc = oscs[1] / abs_max
        other_brt_states=[]
        abs_max=0
        for wlnm in [375,500,700]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        other_brt_states.remove([lmax,abs_max])
        bright_calc=np.float64(evtonm/lmax)
        fratio_calc = oscs[1]/abs_max

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[3] * (fratio_calc - fratio_exp) ** 2 
        self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fratio_exp=fratio_exp, fratio_calc=fratio_calc, fitness=fitness, total_fit=self.fit_iter)
        
        write_gnu(strng, file)
        self.Bright_energy.update({file: [bright_exp, bright_calc]})
        self.D1_energy.update({file: [d1_exp, d1_calc]})


    def optimize_chlorine(self):
        active = self.params[3]
        inactive = [self.params[0], self.params[1], self.params[2]]
        bds = self.bds[12:]

        print(f"\nInitial parameters (active): alpha_cl = {active[0]:.6f}, tccl = {active[1]:.6f}, Uclcl = {active[2]:.6f}, r0clcl = {active[3]:.6f}")
        print(f"\nCarbon parameters (inactive): t = {inactive[0][0]:.6f}, tp = {inactive[0][0] * 2.2 / 2.4:.6f}, U = {inactive[0][1]:.6f}, r0 = {inactive[0][2]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, f_d1/f_bright = {self.weights[2]:.6f}")
        print(f"\nBounds: alpha_cl= {bds[0]}, tccl = {bds[1]}, Uclcl = {bds[2]}, r0clcl = {bds[3]} ")
        
        with open("chlorine_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters: alpha_cl = {active[0]:.6f}, tccl = {active[1]:.6f}, Uclcl = {active[2]:.6f}, r0clcl = {active[3]:.6f}")
            f.write(f"\nCarbon parameters (inactive): t = {inactive[0][0]:.6f}, tp = {inactive[0][0] * 2.2 / 2.4:.6f}, U = {inactive[0][1]:.6f}, r0 = {inactive[0][2]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, f_d1/f_bright = {self.weights[2]:.6f}")
            f.write(f"\nBounds: alpha_cl= {bds[0]}, tccl = {bds[1]}, Uclcl = {bds[2]}, r0clcl = {bds[3]} ")
        
        fnl_rlt = optimize.minimize(self.fitness, active, args=(inactive, self.weights, self.atom), 
                          method='nelder-mead', bounds=bds, 
                          options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshpae(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        

    def optimize_carbon(self):
        active = self.params[0]
        inactive = [self.params[1], self.params[2], self.params[3]]
        bds = self.bds[:3]

        print(f"\nInitial parameters: t = {active[0]:.6f}, tp = (2.2/2.4*t) = {active[0] * 2.2 / 2.4:.6f}, U = {active[1]:.6f}, r0 = {active[2]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        print(f"\nBounds: t = {bds[0]}, U = {bds[1]}, r0 = {bds[2]}")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")

        with open(f"{self.jobname}.txt", "w") as f:
            f.write(f"Execution of ExROPPP parameter fitting code rad_optimise started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters: t = {active[0]:.6f}, tp = (2.2/2.4*t) = {active[0] * 2.2 / 2.4:.6f}, U = {active[1]:.6f}, r0 = {active[2]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            f.write(f"\nBounds: t = {bds[0]}, U = {bds[1]}, r0 = {bds[2]}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")

        
        fnl_rlt = optimize.minimize(
            self.fitness, active, args=(inactive, self.weights, self.atom), 
            method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshpae(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def optimize_pyrrole(self):
        active = self.params[2]
        inactive = [self.params[0], self.params[1], self.params[3]]
        bds = self.bds[8:12]

        print(f"\nInitial parameters (pyrrole): alphan2 = {active[0]:.6f}, tcn2 = {active[1]:.6f}, tpcn2 = (2.2/2.4*tcn2) = {active[1] * 2.2 / 2.4:.6f}, Un2n2 = {active[2]:.6f}, r0n2n2 = {active[3]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        print(f"\nBounds: alphan2= {bds[0]}, tcn2 = {bds[1]}, tpcn2 = {bds[2]}, Un2n2 = {bds[3]} ")
        with open("pyrrole_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters (pyrrole): alphan2 = {active[0]:.6f}, tcn2 = {active[1]:.6f}, tpcn2 = {active[1] * 2.2 / 2.4:.6f}, Un2n2 = {active[2]:.6f}, r0n2n2 = {active[3]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            f.write(f"\nBounds: alphan2= {bds[0]}, tcn2 = {bds[1]}, tpcn2 = {bds[2]}, Un2n2 = {bds[3]} ")
        
        fnl_rlt = optimize.minimize(
            self.fitness, active, args=(inactive, self.weights, self.atom), 
            method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})

        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshpae(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def optimize_pyridine(self):
        active = self.params[1]
        inactive = [self.params[0], self.params[2], self.params[3]]
        bds = self.bds[4:8]

        print(f"\nInitial parameters (pyridine): alphan = {active[0]:.6f}, tcn = {active[1]:.6f}, Unn = {active[2]:.6f}, r0nn = {active[3]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        print(f"\nBounds: alphan= {bds[0]}, tcn = {bds[1]}, Unn = {bds[2]}, r0nn = {bds[3]} ")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")
        
        with open("pyridine_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters (pyridine): alphan = {active[0]:.6f}, tcn = {active[1]:.6f}, Unn = {active[2]:.6f}, r0nn = {active[3]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")

        
        fnl_rlt = optimize.minimize(self.fitness, active, args=(inactive, self.weights, self.atom), 
                                    method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})

        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshpae(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    def optimize_all(self):
        active = np.array(self.params)
        inactive = []
        
        print(f"\nInitial parameters: t = {self.params[0][0]:.6f}, tp = {self.params[0][0] * 2.2 / 2.4:.6f}, U = {self.params[0][1]:.6f}, r0 = {self.params[0][2]:.6f}, "
            f"alphan = {self.params[1][0]:.6f}, tcn = {self.params[1][1]:.6f}, Unn = {self.params[1][2]:.6f}, r0nn = {self.params[1][3]:.6f}, "
            f"alphan2 = {self.params[2][0]:.6f}, tcn2 = {self.params[2][1]:.6f}, tpcn2 = {self.params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {self.params[2][2]:.6f}, r0n2n2 = {self.params[2][3]:.6f}, "
            f"alpha_cl = {self.params[3][0]:.6f}, tccl = {self.params[3][1]:.6f}, Uclcl = {self.params[3][2]:.6f}, r0clcl = {self.params[3][3]:.6f}")
        print(f"\nBounds: t = {self.bds[0]}, U = {self.bds[1]}, r0 = {self.bds[2]}, alphan = {self.bds[4]}, tcn = {self.bds[5]}, Unn = {self.bds[6]}, r0nn = {self.bds[7]}, "
            f"alphan2 = {self.bds[8]}, tcn2 = {self.bds[9]}, Un2n2 = {self.bds[10]}, r0n2n2 = {self.bds[11]}, alpha_cl = {self.bds[12]}, tccl = {self.bds[13]}, Uclcl = {self.bds[14]}, r0clcl = {self.bds[15]}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}, osc ratio (if not C) = {self.weights[3]:.6f}")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")

        with open(f"{self.jobname}.txt", "w") as f:
            f.write(f"Execution of ExROPPP parameter fitting code rad_optimise started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n---------------------- Parameter Fitting  ----------------------\n")
            f.write(f"\nInitial parameters: t = {self.params[0][0]:.6f}, tp = {self.params[0][0] * 2.2 / 2.4:.6f}, U = {self.params[0][1]:.6f}, r0 = {self.params[0][2]:.6f}, "
                    f"alphan = {self.params[1][0]:.6f}, tcn = {self.params[1][1]:.6f}, Unn = {self.params[1][2]:.6f}, r0nn = {self.params[1][3]:.6f}, "
                    f"alphan2 = {self.params[2][0]:.6f}, tcn2 = {self.params[2][1]:.6f}, tpcn2 = {self.params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {self.params[2][2]:.6f}, r0n2n2 = {self.params[2][3]:.6f}, "
                    f"alpha_cl = {self.params[3][0]:.6f}, tccl = {self.params[3][1]:.6f}, Uclcl = {self.params[3][2]:.6f}, r0clcl = {self.params[3][3]:.6f}")
            f.write(f"\nBounds: t = {self.bds[0]}, U = {self.bds[1]}, r0 = {self.bds[2]}, alphan = {self.bds[4]}, tcn = {self.bds[5]}, Unn = {self.bds[6]}, r0nn = {self.bds[7]}, "
            f"alphan2 = {self.bds[8]}, tcn2 = {self.bds[9]}, Un2n2 = {self.bds[10]}, r0n2n2 = {self.bds[11]}, alpha_cl = {self.bds[12]}, tccl = {self.bds[13]}, Uclcl = {self.bds[14]}, r0clcl = {self.bds[15]}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}, osc ratio (if not C) = {self.weights[3]:.6f}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")

        fnl_rlt = optimize.minimize(
            self.fitness, 
            np.array(active), 
            args=(inactive, self.weights, self.atom), 
            method='nelder-mead', 
            bounds=self.bds, 
            options={'maxiter': 10000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6}
        )
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{np.array(fnl_rlt.x).reshape(4, -1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    


now=datetime.now()
print("Execution of ExROPPP parameter fitting code rad_optimise started at: "+now.strftime("%Y-%m-%d %H:%M:%S"))

start_time = time.perf_counter()
optimizer = ParameterOptimization(
        params=weights_file.params,
        weights=weights_file.weights,
        atom='all',
        jobname=args.weights_file,
        normalise_weights=weights_file.normalise_weights,
        hydrocbns=weights_file.hydrocbns,
        heterocyls=weights_file.heterocyls,
        hydcbn_data=weights_file.hydcbn_data,
        hetero_data=weights_file.hetero_data,
        opt=weights_file.opt,
        bounds=bds
    )
optimizer.optimize_params()
print("\nExecution of ExROPPP parameter fitting code rad_optimise finished at: "+now.strftime("%Y-%m-%d %H:%M:%S"))
exec_time = time.perf_counter() - start_time
print('Execution time / h:mm:ss.------: ' +str(exec_time))
