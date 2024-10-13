import numpy as np
import scipy.sparse.linalg as sp
import scipy.linalg as linalg
from datetime import datetime
from rad_settings_beta import *#rad_settings
from subprocess import getoutput
import sys
import os

start_time = datetime.now()

#Function to read geometries, stripping away any non-carbon atoms
#returns nx3 array of xyz coordinates
def read_geom(file):
    print("----------------------------------------------------")
    print("Cartesian Coordinates (as read from file) / Angstrom")
    print("----------------------------------------------------\n")
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
    array_all=np.concatenate((array,array_h))
    atoms=atoms_c+atoms_n+atoms_cl+atoms_h # carbon then nitrogen then chlorine then hydrogen
    print("\n--------------------------------------------------------")
    print("Cartesian Coordinates (in order of atom type) / Angstrom")
    print("--------------------------------------------------------\n")
    for atom in range(array_all.shape[0]):
        print("%s     %f     %f     %f"%(atoms[atom][0],array_all[atom][0],array_all[atom][1],array_all[atom][2]))
    print(" ")
    return array,atoms,array_all,natoms_c,natoms_n,natoms_cl,natoms_c+natoms_n+natoms_cl

#Function to calculate and return interatomic distances
def distance(array):
    dist_array=np.zeros((array.shape[0],array.shape[0]))
    for i in range (np.shape(array)[0]):
        for j in range(i+1,np.shape(array)[0]):
            distance=0
            for k in range (3):
                distance=distance+(array[i,k]-array[j,k])**2
            distance= np.sqrt(distance)
            dist_array[i,j]=distance
            dist_array[j,i]=dist_array[i,j]
    return dist_array

def re_center(coords,atoms,coords_h):
    com = np.zeros(3)
    summass=0
    for i in range(coords_h.shape[0]):
        for x in range(3):
            com[x] += atoms[i][1]*coords_h[i,x]
        summass+=atoms[i][1]
    for x in range(3):
        com[x] = com[x]/summass
    for i in range(coords.shape[0]):
        coords[i,:] -= com
        #return re-centered coords of heavy atoms only (excl. hydrogen) 
    return com,coords

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

# routine to compute dihedral angles between single-bonded groups (i.e. two rings) for angle-dependent hopping terms 
def dihedrals(natoms,atoms,coords,dist_array):
    a2,bond_list=adjacency(dist_array,cutoff) #adjacency
    a3=np.dot(a2,a2) # 1-3 interactions
    a4=np.dot(a3,a2) # 1-4 interactions
    lst=[]
    for i in range(natoms):
        for j in range(i+1,natoms):
            if a4[i,j]!=0 and a3[i,j]==0 and a2[i,j]==0: #if is non zero in a4 but zero in a3 and a2 then is a true 1-4 interaction (dihedral)
                lst.append([i,j])
                lst.append([j,i])
    angles={}
    for dihedral in lst:
        for bond in bond_list:
            if a2[dihedral[0],bond[0]]==1 and a2[dihedral[1],bond[1]]==1:
                if atoms[bond[0]][0] in ['C','c'] and atoms[bond[1]][0] in ['C','c']:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta)
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
                elif array_intersect([atoms[bond[0]][0],atoms[bond[1]][0]],['N','n','N2','n2']) in [['N'],['n'],['N2'],['n2']]:
                    theta=compute_angle([dihedral[0],bond[0],bond[1],dihedral[1]],coords)
                    if '%s-%s'%(bond[0],bond[1]) in angles:
                        angles['%s-%s'%(bond[0],bond[1])].append(theta)
                    else:
                        angles.update({'%s-%s'%(bond[0],bond[1]):[theta]})
    for bond in angles: # average over dihedral angles between same two groups to account for slight non-planarity of rings
        avg_angle=sum(angles[bond])/len(angles[bond])
        angles.update({bond:avg_angle})
    print(angles)
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

#Optional function to flip signs of orbital coeffs such that for every pair of bonding-antibonding orbitals, 
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
def t_term(dist_array,natoms_c,natoms_n,natoms,n_list,theta):
    array=np.zeros_like(dist_array)
    # C-C hopping 
    ntheta=0
    for i in range (natoms_c):
        for j in range (i+1,natoms_c):
            if dist_array[i,j]<cutoff:
                if '%s-%s'%(i,j) in theta:
                    #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                    array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*A*np.exp(-b*dist_array[i,j])
                    ntheta+=1
                else:
                    array[i,j]=A*np.exp(-b*dist_array[i,j])
                array[j,i]=array[i,j]  
                #ntheta+=1
    # N and Cl hopping 
    # C-N hopping
    for i in range (natoms_c):
        for j in range (natoms_c,natoms_c+natoms_n):
            if dist_array[i,j]<cutoff:
                if n_list[j-natoms_c]==0:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn*np.exp(-bcn*dist_array[i,j])
                        ntheta+=1
                    else:
                       array[i,j]= Acn*np.exp(-bcn*dist_array[i,j])
                    print("C-N1 bond")
                elif n_list[j-natoms_c]==1:
                    if '%s-%s'%(i,j) in theta:
                        #print("Used Theta %d %f deg. atoms %d %d"%(ntheta,theta['%s-%s'%(i,j)],i+1,j+1))
                        array[i,j]=abs(np.cos(np.pi*theta['%s-%s'%(i,j)]/180))*Acn2*np.exp(-bcn2*dist_array[i,j])  
                        ntheta+=1
                    else:
                        array[i,j]=Acn2*np.exp(-bcn2*dist_array[i,j]) 
                    print("C-N2 bond")
                array[j,i] = array[i,j]
    # N-N hopping 
   # for i in range (natoms_c,natoms_c+natoms_n):
    #    for j in range (i+1,natoms_c+natoms_n):
       #     if dist_array[i,j]<cutoff:
       #         if n_list[i-natoms_c]==0 and n_list[j-natoms_c]==0:
          #          array[i,j] = tnn
            #        print("N1-N1 bond")
             #   elif n_list[i-natoms_c]+n_list[j-natoms_c]==1:
              #      array[i,j] = tnn2
              #      print("N1-N2 bond")
              #  elif n_list[i-natoms_c]==1 and n_list[j-natoms_c]==1:
               #     array[i,j] = tn2n2
               #     print("N2-N2 bond")
               # array[j,i] = array[i,j]  
    # C-Cl hopping
    for i in range (natoms_c):
        for j in range (natoms_c+natoms_n,natoms):
            if dist_array[i,j]<ccl_cutoff:
                array[i,j]=Accl*np.exp(-bccl*dist_array[i,j])
                array[j,i]=array[i,j]  
                ntheta+=1
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
def v_term(dist_array,natoms_c,natoms_n,natoms,n_list):  
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
    for l in range(orbs.shape[0]): # atom l
        for m in range(orbs.shape[0]): # atom m
            J00 += orbs[l,ndocc]**2 * orbs[m,ndocc]**2 * repulsion[l,m]
    return J00

#Function to calculate open-shell SCF energy
def energy(hopping,repulsion,fock_mat,density,orbs,ndocc):
    J00 = compute_j00(orbs,repulsion,ndocc)
    return 0.5*(np.dot(density.flatten(),hopping.flatten())+np.dot(density.flatten(),fock_mat.flatten())) - 0.25*J00

#Main HF function
def main_scf(file):
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
    hopping=t_term(dist_array,natoms_c,natoms_n,natoms,n_list,angles)
    repulsion=v_term(dist_array,natoms_c,natoms_n,natoms,n_list)
#Diagonalize Huckel Hamiltonian to form initial density guess
    guess_evals,evecs=np.linalg.eigh(hopping)
    guess_dens=density(evecs,natoms,ndocc)
    #nrise=0
    nrise=1
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

def write_gnu(strng):
    f=open('gnuplot_script','w')
    f.write("#simulated spectrum\n")
    f.write("set term pdf size 6,4\n")
    f.write("unset key\n")
    f.write("set output '%s.pdf'\n" %(file))
    f.write("set xrange [200:700]\n")
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units'\n")
    f.write("p %s lw 2 dt 1" %strng)
    f.close()
    return

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
    f.close()
    print("done!")
    return  

def spin(ndocc,norbs,cis_coeffs,nstates):
    spinmat = np.zeros((nstates,nstates))   
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

def dipole(coords,atoms,norbs,hforbs,ndocc,nstates,basis):
    print("Calculating dipole moments ...\n")
    # Routine to calculate the one electron dipole moment matrix (x, y and z) 
    # in the basis of orbitals, and then the dipole moment matrix in the basis
    # of excitations/CSFs
    natoms = coords.shape[0] 
    dip1el = np.zeros((norbs,norbs,3))
    for i in range(norbs):
        for j in range(i,norbs):
            for u in range(natoms):
                for x in range(3):
                    dip1el[i,j,x] += hforbs[u,i]*coords[u,x]*hforbs[u,j]*tobohr
                    dip1el[j,i,x] = dip1el[i,j,x]
    dipoles = np.zeros((nstates,nstates,3)) 
            
    if basis=='xct':
        nunocc = norbs-ndocc-1
        #1 <0|mu|0>
        for x in range(3):
            for m in range(ndocc):
                dipoles[0,0,x] -= 2*dip1el[m,m,x]
            dipoles[0,0,x] -= dip1el[ndocc,ndocc,x]
        #2 <0|mu|ibar->0bar> 
        for x in range(3):
            for i in range(ndocc):
                dipoles[0,i+1,x] = -dip1el[i,ndocc,x]
                dipoles[i+1,0,x] = dipoles[0,i+1,x] 
        #3 <0|mu|0->j'>
        for x in range(3):
            for j in range (nunocc):
                dipoles[0,j+ndocc+1,x] = -dip1el[ndocc,j+ndocc+1,x]
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
            #7 <kbar->0bar|mu|ibar->0bar> 
            for x in range(3):
                for i in range(ndocc):
                    for k in range(ndocc):
                        dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
                        if i==k:
                            dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[ndocc,ndocc,x]
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|i->j'> = 0
            #10 <kbar->0bar|mu|ibar->jbar'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -dip1el[ndocc,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]  
            #11 <kbar->0bar|mu|ibar->0bar,0->j'>
            for x in range(3):
               for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*(ndocc*nunocc)+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -dip1el[ndocc,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            #12 <0->l'|mu|0->j'> 
            for x in range(3):
                for j in range(nunocc):
                    for l in range(nunocc):
                        dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[ndocc,ndocc,x]
            #13 <0->l'|mu|i->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = dip1el[i,ndocc,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #14 <0->l'|mu|ibar->jbar'> = 0
            #15 <0->l'|mu|ibar->0bar,0->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*(ndocc*nunocc)+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = -dip1el[i,ndocc,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #16 <k->l'|mu|i->j'>  
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
            #19 <kbar->lbar'|mu|ibar->jbar'>  
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
            #21 <kbar->0bar,0->l'|mu|ibar->0bar,0->j'>  
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
    
    if basis == 'rot':
        nunocc = norbs-ndocc-1
        #1 <0|mu|0>
        for x in range(3):
            for m in range(ndocc):
                dipoles[0,0,x] -= 2*dip1el[m,m,x]
            dipoles[0,0,x] -= dip1el[ndocc,ndocc,x]
        #2 <0|mu|ibar->0bar> 
        for x in range(3):
            for i in range(ndocc):
                dipoles[0,i+1,x] = -dip1el[i,ndocc,x]
                dipoles[i+1,0,x] = dipoles[0,i+1,x] 
        #3 <0|mu|0->j'>
        for x in range(3):
            for j in range (nunocc):
                dipoles[0,j+ndocc+1,x] = -dip1el[ndocc,j+ndocc+1,x]
                dipoles[j+ndocc+1,0,x] = dipoles[0,j+ndocc+1,x]
        #4 <0|mu|4,i->j'>=0
        #5 <0|mu|2S,i->j'>
        for x in range(3):
            for n in range (ndocc*nunocc):
                nn = n+ndocc+nunocc+ndocc*nunocc+1
                i = int(np.floor(n/nunocc))
                j = n-i*nunocc+ndocc +1
                dipoles[0,nn,x] = -np.sqrt(2)*dip1el[i,j,x]
                dipoles[nn,0,x] = dipoles[0,nn,x]
        #6 <0|mu|2T,i->j'>=0
        if mixing==True:
            print("Dipole moments are corrected for ground state mixing of excited configurations in rotated basis.\n")
            #7 <kbar->0bar|mu|ibar->0bar>  
            for x in range(3):
                for i in range(ndocc):
                    for k in range(ndocc):
                        dipoles[i+1,k+1,x] = +dip1el[i,k,x] 
                        if i==k:
                            dipoles[i+1,k+1,x] += dipoles[0,0,x] - dip1el[ndocc,ndocc,x]
            #8 <kbar->0bar|mu|0->j'> = 0
            #9 <kbar->0bar|mu|4,i->j'>=0
            #10 <kbar->0bar|mu|2S,i->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -1/np.sqrt(2)*dip1el[ndocc,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]
            #11 <kbar->0bar|mu|2T,i->j'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[i+1,nn,x] = -3/np.sqrt(6)*dip1el[ndocc,j,x]
                    dipoles[nn,i+1,x] = dipoles[i+1,nn,x]    
            #12 <0->j'|mu|0->l'>  
            for x in range(3):
                for j in range(nunocc):
                    for l in range(nunocc):
                        dipoles[j+ndocc+1,l+ndocc+1,x] = -dip1el[j+ndocc+1,l+ndocc+1,x]
                        if j==l:
                           dipoles[j+ndocc+1,l+ndocc+1,x] += dipoles[0,0,x] + dip1el[ndocc,ndocc,x]
            #13 <0->j'|mu|4,k->l'> = 0
            #14 <0->j'|mu|2S,k->l'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = 1/np.sqrt(2)*dip1el[i,ndocc,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #15 <0->j'|mu|2T,k->l'>
            for x in range(3):
                for n in range(ndocc*nunocc):
                    nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                    i = int(np.floor(n/nunocc))
                    j = n-i*nunocc+ndocc +1
                    dipoles[j,nn,x] = -3/np.sqrt(6)*dip1el[i,ndocc,x]
                    dipoles[nn,j,x] = dipoles[j,nn,x]
            #16 <4,i->j'|mu|4,k->l'>  
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
            #17 <4,k->l'|mu|2S,i->j'> = 0
            #18 <4,k->l'|mu|2T,i->j'> = 0
            #19 <2S,i->j'|mu|2S,k->l'>  
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
            #20 <2S,i->j'|mu|2T,k->l'> = 0 
            
            #21 <2T,i->j'|mu|2T,k->l'>
            for x in range(3):
                 for m in range(ndocc*nunocc):
                     mm = m+ndocc+nunocc+2*ndocc*nunocc+1
                     k = int(np.floor(m/nunocc))
                     l = m-k*nunocc+ndocc +1
                     for n in range (ndocc*nunocc):
                         nn = n+ndocc+nunocc+2*ndocc*nunocc+1
                         i = int(np.floor(n/nunocc))
                         j = n-i*nunocc+ndocc +1
                         if i==k:
                             dipoles[mm,nn,x] = -dip1el[j,l,x]
                         if j==l:
                             dipoles[mm,nn,x] += dip1el[i,k,x]
                         if i==k and j==l:
                             dipoles[mm,nn,x] += dipoles[0,0,x]
    perm_dip=dipoles[0,0,:]
    for i in range(norbs):
        atom_z=0
        if atoms[i][0] in ['C','c','n1','N1']:
            atom_z=1
        elif atoms[i][0] in ['Cl','cl','CL','N2','n2']:
            atom_z=2   
        for x in range(3):
            perm_dip[x]+=atom_z*coords[i,x]*tobohr
    print("Permanent dipole moment of ground state: mu0 = %7.3f x %7.3f y %7.3f z\n"%(perm_dip[0],perm_dip[1],perm_dip[2]))
    return dipoles
    
def hetero_cisd_ham(ndocc,norbs,energy0,orb_energies,rep_tens):
    nunocc = norbs-ndocc-1
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    cish = np.zeros((nstates,nstates))
    #cish[0:nelec,0:nelec] = cis_ham_sml
    #1 <0|H|0>
    cish[0,0] = energy0
    #2 <0|H|ibar->0bar> 
    for i in range (ndocc): 
        cish[0,i+1] = 0.5*rep_tens[i,ndocc,ndocc,ndocc] 
        cish[i+1,0] = cish[0,i+1]
    #3 <0|H|0->j'>
    for j in range (nunocc):
        cish[0,j+ndocc+1] = -0.5*rep_tens[ndocc,j+ndocc+1,ndocc,ndocc] 
        cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    #4 <0|H|i->j'> # 
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        #print(i,j)
        cish[0,nn] = -0.5*rep_tens[i,ndocc,ndocc,j]
        cish[nn,0] = cish[0,nn]
    #5 <0|H|ibar->jbar'> #  
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = 0.5*rep_tens[i,ndocc,ndocc,j]
        cish[nn,0] = cish[0,nn]
    #6 <0|H|ibar->0bar,0->j'> #
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+2*ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = rep_tens[i,ndocc,ndocc,j]
        cish[nn,0] = cish[0,nn]
    #7 <kbar->0bar|H|ibar->0bar>  
    for i in range(ndocc):
        for k in range(ndocc):
            cish[i+1,k+1] = -rep_tens[i,k,ndocc,ndocc] +0.5*rep_tens[i,ndocc,ndocc,k]
            if i==k:
                cish[i+1,i+1] += energy0 + orb_energies[ndocc] - orb_energies[i] + 0.5*rep_tens[ndocc,ndocc,ndocc,ndocc]
    #8 <kbar->0bar|H|0->j'>
    for j in range(nunocc):
        for k in range(ndocc):
            cish[k+1,j+ndocc+1] = rep_tens[ndocc,k,ndocc,j+ndocc+1] 
            cish[j+ndocc+1,k+1] = cish[k+1,j+ndocc+1]
    #9 <kbar->0bar|H|i->j'> 
    for k in range(ndocc):    
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = rep_tens[ndocc,k,i,j]
            cish[nn,k+1] = cish[k+1 ,nn]          
    #10 <kbar->0bar|H|ibar->jbar'> 
    for k in range(ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = rep_tens[ndocc,k,i,j] - rep_tens[ndocc,j,i,k] #10a) and 10b)
            if i==k:
                cish[k+1,nn] += 0.5*rep_tens[ndocc,j,ndocc,ndocc] #10a)
            cish[nn,k+1] = cish[k+1,nn]
    #11 <kbar->0bar|H|ibar->0bar,0->j'> 
    for k in range (ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[k+1,nn] = -rep_tens[ndocc,j,i,k] #11a) and 11b)
            if i==k:
                cish[k+1,nn] += 0.5*rep_tens[ndocc,j,ndocc,ndocc] #11a)
            cish[nn,k+1] = cish[k+1,nn]   
    #12 <0->l'|H|0->j'>  
    for j in range(nunocc):
        for l in range(nunocc):
            cish[j+ndocc+1,l+ndocc+1] = - rep_tens[j+ndocc+1,l+ndocc+1,ndocc,ndocc] + 0.5*rep_tens[j+ndocc+1,ndocc,ndocc,l+ndocc+1]
            if j==l:
                cish[j+ndocc+1,j+ndocc+1] += energy0 + orb_energies[j+ndocc+1] - orb_energies[ndocc] + 0.5*rep_tens[ndocc,ndocc,ndocc,ndocc]
    # 13 <0->l'|H|i->j'>
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,j,l,ndocc] - rep_tens[i,ndocc,l,j]
            if l==j: 
                cish[l,nn] += 0.5*rep_tens[i,ndocc,ndocc,ndocc]
            cish[nn,l] = cish[l,nn]  
    #14 <0->l'|H|ibar->jbar'>
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,j,l,ndocc]
            cish[nn,l] = cish[l,nn]
    #15 <0->l'|H|ibar->0bar,0->j'>  
    for l in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            j = n-i*nunocc+ndocc +1
            cish[l,nn] = rep_tens[i,ndocc,l,j]
            if l==j:
              cish[l,nn] += -0.5*rep_tens[i,ndocc,ndocc,ndocc] #15a)
            cish[nn,l] = cish[l,nn]    
    #16 <k->l'|H|i->j'>  
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
                cish[mm,nn] -= 0.5*rep_tens[l,ndocc,ndocc,j] #16b
            if j==l:
                cish[mm,nn] += 0.5*rep_tens[i,ndocc,ndocc,k] #16c
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
                cish[mm,nn] = -rep_tens[ndocc,k,i,ndocc]
                cish[nn,mm] = cish[mm,nn]           
    #19 <kbar->lbar'|H|ibar->jbar'> 
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
                cish[mm,nn] += 0.5*rep_tens[l,ndocc,ndocc,j] 
            if j==l:
                cish[mm,nn] -= 0.5*rep_tens[i,ndocc,ndocc,k]
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
                cish[mm,nn] = rep_tens[l,ndocc,ndocc,j]
            cish[nn,mm] = cish[mm,nn]       
    #21 <kbar->0bar,0->l'|H|ibar->0bar,0->j'> 
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
                cish[mm,nn] += 0.5*rep_tens[l,ndocc,ndocc,j]
            if j==l:
                cish[mm,nn] += 0.5*rep_tens[i,ndocc,ndocc,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 - orb_energies[i] + orb_energies[j]
    # #sys.exit()  
    return cish
            
def hetero_xcis(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs):
    print("")
    print("----------------------------")
    print("Starting ExROPPP calculation for heterocycle monoradical in excitations basis")
    print("----------------------------\n")
    # Transform 2-el ingrls into mo basis
    rep_tens = transform(repulsion,hf_orbs)
    # Construct and diagonalise CIS Hamiltonian for first 25 excited states
    cis_ham_het = hetero_cisd_ham(ndocc,norbs,energy0,orb_energies,rep_tens)
    np.savetxt('big_ham_benz.csv', cis_ham_het, delimiter=',') 
    nunocc = norbs-ndocc-1
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    if states_cutoff_option == 'states' and states_to_print <= nstates:
        rng = states_to_print
        print('Lowest %d states. WARNING -  Any higher lying states will not be included in spectrum.\n'%states_to_print)
    else:
        rng = nstates
    if states_cutoff_option == 'energy':
        cutoff_energy = energy_cutoff
        print('Used energy cutoff of %04.2f eV for states. WARNING - Any higher lying states will not be included in spectrum.\n'%cutoff_energy)
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
    s_squared,deltassq = spin(ndocc,norbs,cis_coeffs,nstates)
    # Calculate dipole moment array
    dip_array = dipole(coords,atoms,norbs,hf_orbs,ndocc,nstates,'xct')
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
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,0]*tobohr*hf_orbs[k,ndocc] #
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,1]*tobohr*hf_orbs[k,ndocc]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,iorb]*coords[k,2]*tobohr*hf_orbs[k,ndocc] 
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
                    tdmx = tdmx + cis_coeffs[j,i]*hf_orbs[k,ndocc]*coords[k,0]*tobohr*hf_orbs[k,jorb] 
                    tdmy = tdmy + cis_coeffs[j,i]*hf_orbs[k,ndocc]*coords[k,1]*tobohr*hf_orbs[k,jorb]
                    tdmz = tdmz + cis_coeffs[j,i]*hf_orbs[k,ndocc]*coords[k,2]*tobohr*hf_orbs[k,jorb] 
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
    write_gnu(strng)
    return strng,cis_energies - cis_energies[0],osc_array

def hetero_ham_rot(ndocc,norbs,energy0,orb_energies,rep_tens):
    nunocc = norbs-ndocc-1
    ndocc=ndocc
    nstates = 3*ndocc*nunocc +ndocc+nunocc +1
    cish = np.zeros((nstates,nstates))
    #1 <0|H|0>
    cish[0,0] = energy0
    #2 <0|H|ibar->0bar> 
    for i in range (ndocc): 
        cish[0,i+1] = 0.5*rep_tens[i,ndocc,ndocc,ndocc] 
        cish[i+1,0] = cish[0,i+1]
    #3 <0|H|0->j'>
    for j in range (nunocc):
        cish[0,j+ndocc+1] = -0.5*rep_tens[ndocc,j+ndocc+1,ndocc,ndocc] 
        cish[j+ndocc+1,0] = cish[0,j+ndocc+1]
    #4 <0|H|Q i->j'> ALL ZERO  
    #5 <0|H|D(+)i->j'> ALL ZERO (only depends on Fij')
    #6 <0|H|D(-)i->j'>
    for n in range (ndocc*nunocc):
        nn = n+ndocc+nunocc+2*ndocc*nunocc+1
        i = int(np.floor(n/nunocc))
        j = n-i*nunocc+ndocc +1
        cish[0,nn] = 3/(np.sqrt(6)) *rep_tens[i,ndocc,ndocc,j]
        cish[nn,0] = cish[0,nn]
    #7 <kbar->0bar|H|ibar->0bar> 
    for i in range(ndocc):
        for k in range(i,ndocc):
            cish[i+1,k+1] = -rep_tens[i,k,ndocc,ndocc] +0.5*rep_tens[i,ndocc,ndocc,k]
            if i==k:
                cish[i+1,i+1] += energy0 + orb_energies[ndocc] - orb_energies[i] + 0.5*rep_tens[ndocc,ndocc,ndocc,ndocc]
            cish[k+1,i+1] = cish[i+1,k+1]
    #8 <kbar->0bar|H|0->j'>
    for j in range(nunocc):
        for k in range(ndocc):
            cish[k+1,j+ndocc+1] = rep_tens[ndocc,k,ndocc,j+ndocc+1]
            cish[j+ndocc+1,k+1] = cish[k+1,j+ndocc+1]
    #9 <0->l',+|H|Q i->j'> ALL ZERO
    #10 <kbar->0bar|H|D(+)i->j'>  
    for k in range(ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[k+1,nn] = np.sqrt(2)*rep_tens[ndocc,k,i,jp] - 1/np.sqrt(2)*rep_tens[ndocc,jp,i,k]
            if i==k:
                cish[k+1,nn] += 1/(2*np.sqrt(2))*rep_tens[ndocc,jp,ndocc,ndocc]
            cish[nn,k+1] = cish[k+1,nn]
    #11 <kbar->0bar|H|D(-)i->j'>
    for k in range (ndocc):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[k+1,nn] = -3/np.sqrt(6)*rep_tens[ndocc,jp,i,k]
            if i==k:
               cish[k+1,nn] += 3/(2*np.sqrt(6))*rep_tens[ndocc,jp,ndocc,ndocc] 
            cish[nn,k+1] = cish[k+1,nn]
    #12 <0->l'|H|0->j'>
    for j in range(nunocc):
        for l in range(j,nunocc):
            cish[j+ndocc+1,l+ndocc+1] = - rep_tens[j+ndocc+1,l+ndocc+1,ndocc,ndocc] + 0.5*rep_tens[j+ndocc+1,ndocc,ndocc,l+ndocc+1]
            if j==l:
                cish[j+ndocc+1,j+ndocc+1] += energy0 + orb_energies[j+ndocc+1] - orb_energies[ndocc] + 0.5*rep_tens[ndocc,ndocc,ndocc,ndocc]
            cish[l+ndocc+1,j+ndocc+1] = cish[j+ndocc+1,l+ndocc+1]
    #13 <0->l',-|H|Q i->j'> ALL ZERO         
    #14 <0->l'|H|D(+)i->j'> 
    for lp in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[lp,nn] = np.sqrt(2)*rep_tens[i,jp,lp,ndocc] -1/np.sqrt(2)*rep_tens[i,ndocc,lp,jp]
            if lp==jp:
                cish[lp,nn] += 1/(2*np.sqrt(2))*rep_tens[i,ndocc,ndocc,ndocc]
            cish[nn,lp] = cish[lp,nn]
    #15 <0->l'|H|D(-)i->j'>
    for lp in range(ndocc+1,ndocc+nunocc+1):
        for n in range (ndocc*nunocc):
            nn = n+ndocc+nunocc+2*ndocc*nunocc+1
            i = int(np.floor(n/nunocc))
            jp = n-i*nunocc+ndocc +1
            cish[lp,nn] = 3/np.sqrt(6)*rep_tens[i,ndocc,lp,jp]
            if jp==lp:
                cish[lp,nn] -= 3/(2*np.sqrt(6))*rep_tens[i,ndocc,ndocc,ndocc]
            cish[nn,lp] = cish[lp,nn]
    #16 <Qk->l'|H|Qi->j'>  
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
                cish[mm,nn] -= 0.5*rep_tens[j,ndocc,ndocc,l]
            if j==l:
                cish[mm,nn] -= 0.5*rep_tens[ndocc,k,i,ndocc]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]        
    #17 <Qk->l'|H|D(+)i->j'> ALL ZERO           
    #18 <Qk->l'|H|D(-)i->j'> ALL ZERO
    #19 <D(+)k->l'|H|D(+)i->j'> 
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
                cish[mm,nn] = 0.5*np.sqrt(3)*rep_tens[j,ndocc,ndocc,l]
            if j==l:
                cish[mm,nn] -= 0.5*np.sqrt(3)*rep_tens[ndocc,k,i,ndocc]
            cish[nn,mm] = cish[mm,nn] 
    #21 <D(-)k->l'|H|D(-)i->j'> 
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
                cish[mm,nn] += rep_tens[j,ndocc,ndocc,l]
            if j==l:
                cish[mm,nn] += rep_tens[i,ndocc,ndocc,k]
            if i==k and j==l:
                cish[mm,nn] += energy0 + orb_energies[j] - orb_energies[i]
            cish[nn,mm] = cish[mm,nn]
    return cish

def hetero_xcis_rot(ndocc,norbs,coords,atoms,energy0,repulsion,orb_energies,hf_orbs):
    print("")
    print("----------------------------")
    print("Starting ExROPPP calculation for monoradical heterocycle in rotated basis")
    print("----------------------------\n")
    # Transform 2-el ingrls into mo basis
    rep_tens = transform(repulsion,hf_orbs)
    # Construct CIS Hamiltonian
    het_ham_rot = hetero_ham_rot(ndocc,norbs,energy0,orb_energies,rep_tens)
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
    dip_array = dipole(coords,atoms,norbs,hf_orbs,ndocc,nstates,'rot')
    aku=np.einsum("ijx,jk",dip_array,cis_coeffs)
    mu0u=np.einsum("j,jix",cis_coeffs[:,0].T,aku)
    osc_array=[]
    print("Ground state energy relative to E(|0>): %04.3f eV\n"%(cis_energies[0]-energy0))
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
    strng = strng[1:]
    write_gnu(strng)     
    return strng, cis_energies-cis_energies[0],osc_array

def rad_calc(file):
    #Call HF driver--Comment out last line to stop here
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,one_body,two_body,orb_energy,hf_orbs,fock_mat=main_scf(file)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    hf_orbs = orb_sign(hf_orbs,orb_energy,nelec,dist_array,alt)
    # print relevant orbs
    print("\n--------------------------")
    print("Converged ROPPP Orbitals")
    print("--------------------------\n")
    natoms=np.shape(coord)[0]
    for iorb in range(natoms):
        print('orbital number', iorb + 1, 'energy', orb_energy[iorb]-orb_energy[int((nelec-1)/2)])
        print(np.around(hf_orbs[:, iorb], decimals=2)) 
    
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
            if i!=j and dens_mo[i,j] > 1e-4:
                print("Density matrix not converged!")
                print("\nDensity Matrix:")
                print(dens_mo)
                sys.exit()

    strng,ci_energies_array,osc_array = hetero_xcis_rot(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs)
    return strng,ci_energies_array,osc_array  #return gnuplot data for plotting spectrum

# ----------- Execute radicals code -----------   
now=datetime.now()
print("Execution of ExROPPP code rad7 started at: "+now.strftime("%Y-%m-%d %H:%M:%S"))
print()
strng,energy_array,osc_array = rad_calc(file)

now=datetime.now()
print("\nExecution of ExROPPP code rad7 finished at: "+now.strftime("%Y-%m-%d %H:%M:%S"))
exec_time = datetime.now() - start_time
print('Execution time / h:mm:ss.------: ' +str(exec_time))