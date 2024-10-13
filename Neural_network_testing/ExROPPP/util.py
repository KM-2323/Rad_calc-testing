from rad_settings_opt import *
import os



import numpy as np
import scipy.linalg as linalg

def read_geom(file):
    print("--------------------------------")
    print("Cartesian Coordinates / Angstrom")
    print("--------------------------------\n")
    molecules_folder = 'Molecules'
    file_path = os.path.join(molecules_folder, file)
    f=open(file_path,'r')
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
        if splt_ln[0] in ["Cl","cl", 'CL']:
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
