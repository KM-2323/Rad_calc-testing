import numpy as np
import os
import re
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
import numba as nb




# Get the coolwarm colormap
coolwarm = plt.get_cmap('coolwarm')
# Create a new colormap with only the cool part (0.0 to 0.5)
cool_cmap = LinearSegmentedColormap.from_list('cool', coolwarm(np.linspace(0, 0.3, 256)))
warm_cmap = LinearSegmentedColormap.from_list('warm', coolwarm(np.linspace(0.7, 1, 256)))
class MolecularOrbitals:
    """
    A class to represent molecular orbitals extracted from an output file.

    Attributes
    ----------
    output_file_path : str
        Path to the output file containing molecular orbital data.
    coordinates : dict
        Dictionary containing atomic coordinates.
    eigenvectors : list
        List of eigenvectors extracted from the output file.
    basis : list
        List of basis coordinates for non-zero eigenvectors.
    non_zero_eigenvectors : list
        List of non-zero eigenvectors.
    mos : dict
        Dictionary containing molecular orbitals.
    num : int
        Number of molecular orbitals.

    Methods
    -------
    mos_num() -> int:
        Returns the number of molecular orbitals.
    extract_coordinate() -> dict:
        Extracts and returns a dictionary of atomic coordinates from the output file.
    extract_eigenvectors() -> list:
        Extracts and returns a list of eigenvectors from the output file.
    extract_basis() -> tuple[list, list]:
        Extracts and returns basis coordinates and non-zero eigenvectors.
    mo_dict() -> dict:
        Creates and returns a dictionary of molecular orbitals.
    """

    def __init__(self, output_file_path, compound_name):
        """
        Constructs all the necessary attributes for the MolecularOrbitals object.

        Parameters
        ----------
        output_file_path : str
            Path to the output file containing molecular orbital data.
        """
        if os.path.exists(output_file_path):
            self.output_file_path = output_file_path
        else:
            raise FileNotFoundError(f"The file {output_file_path} does not exist.")
        
        self.coordinates = self.extract_coordinate()
        self.eigenvectors = self.extract_eigenvectors()
        self.basis, self.non_zero_eigenvectors = self.extract_basis()
        self.mos = self.mo_dict()
        self.num = self.mos_num()
        self.name = compound_name
        self.somo_index = self.somo_getter()

    def somo_getter(self) -> str:
        return f'MO{(self.num + 1) / 2}'

    def mos_num(self) -> int:
        """
        Returns the number of molecular orbitals.

        Returns
        -------
        int
            Number of molecular orbitals.
        """
        return len(self.mos)

    def extract_coordinate(self) -> dict:
        """
        Extracts and returns a dictionary of atomic coordinates from the output file.

        Returns
        -------
        dict
            Dictionary where each key (element) has value (coordinate) in a list.
        """
        with open(self.output_file_path, 'r') as read_mos:
            all_data_str = read_mos.read()

        counter = 1
        coordinates = {}

        coord_pattern_C = re.compile(r'C\s+6\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N = re.compile(r'N\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_N2 = re.compile(r'N2\s+7\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')
        coord_pattern_Cl = re.compile(r'Cl\s+17\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)')

        for match in coord_pattern_C.finditer(all_data_str):
            coordinates[f'C{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_N.finditer(all_data_str):
            coordinates[f'N{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_N2.finditer(all_data_str):
            coordinates[f'N{counter}'] = [float(match.group(1)), 
                                          float(match.group(2)), 
                                          float(match.group(3))]
            counter += 1

        for match in coord_pattern_Cl.finditer(all_data_str):
            coordinates[f'Cl{counter}'] = [float(match.group(1)), 
                                           float(match.group(2)), 
                                           float(match.group(3))]
            counter += 1
        
        return coordinates

    def extract_eigenvectors(self) -> list:
        """
        Extracts and returns all the eigenvectors from the output file.

        Returns
        -------
        list
            List of eigenvectors.
        """
        eigenvectors = []
        collecting = False
        with open(self.output_file_path, 'r') as f2:
            all_data_lines = f2.readlines()
        line_pattern = re.compile(r'\s*(\d+)\s+([A-Z][a-zA-Z]?\s*\d+)\s+([SXYZ])\s+([-.\d]+)\s*')
        for line in all_data_lines:
            if "EIGENVECTORS" in line:
                collecting = True
                coefficients = []
            elif "...... END OF ROHF CALCULATION ......" in line:
                if collecting:
                    eigenvectors.append(coefficients)
                    collecting = False
            elif collecting:
                match = line_pattern.match(line)
                if match:
                    atom_info = re.sub(r'\s+', '', match.group(2))  # Remove any extra spaces
                    formatted_basis = [atom_info, match.group(3), float(match.group(4))]
                    coefficients.append(formatted_basis)
        
        return eigenvectors

    def extract_basis(self) -> tuple[list, list]:
        """
        Extracts and returns basis coordinates and non-zero eigenvectors.

        Returns
        -------
        tuple
            Two lists: non-zero eigenvectors and basis coordinates.
        """
        basis = []
        non_zero_eigenvectors = []
        counter = 0
        for i in range(len(self.eigenvectors)):
            basis.append([self.coordinates.get(eigenvector[0].strip()) for eigenvector in self.eigenvectors[i] if eigenvector[-1] != 0.0])
            
            non_zero_eigenvectors.append([eigenvector for eigenvector in self.eigenvectors[i] if eigenvector[-1] != 0.0])
            counter += 1
        # print(basis)
        return basis, non_zero_eigenvectors

    def mo_dict(self) -> dict:
        """
        Creates and returns a dictionary of molecular orbitals.

        Returns
        -------
        dict
            Dictionary of molecular orbitals, each composed of an array of non-zero n_atoms * 4 (x,y,z,coef).
        """
        mos = {}
        counter = 1        
        # print(self.non_zero_eigenvectors)
        for bas, vec in zip(self.basis, self.non_zero_eigenvectors):
            for_plot = []
            for i in range(len(bas)):
                basis_coef = float(np.array(vec)[i,-1])
                
                coord = np.array(bas)[i].astype(np.float64)
                combined = np.append(coord, basis_coef)
                for_plot.append(combined)
            mos[f'MO{counter}'] = np.array(for_plot)
            counter += 1     
        return mos

class MODrawer(MolecularOrbitals):
    """
    A class to draw and visualize molecular orbitals.

    Methods
    -------
    set_axes_equal(ax):
        Sets equal scaling for 3D plot axes.
    mo_drawer(key, orbital_scale=2, viewing_angle=[45, -90], figsize=(14, 10), transparency=0.6, bondthickness=2, savefig=False):
        Draws the molecular orbital specified by the key.
    """
    def __init__(self, output_file_path, compound_name):
        super().__init__(output_file_path, compound_name)
    
    @staticmethod
    def generate_sphere_mesh(center, radius, resolution=20):
        """
        Generates vertices and faces to approximate a sphere.
        
        Parameters:
        - center: A tuple or list with the (x, y, z) coordinates of the sphere center.
        - radius: The radius of the sphere.
        - resolution: The number of subdivisions in the theta and phi directions.
        
        Returns:
        - vertices: A list of vertices (each a list of [x, y, z]).
        - faces: A list of faces (each a list of 3 or 4 vertex indices).
        """
        vertices = []
        faces = []
        
        
        # Generate vertices
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        
        for t in theta:
            for p in phi:
                x = radius * np.sin(t) * np.cos(p) + center[0]
                y = radius * np.sin(t) * np.sin(p) + center[1]
                z = radius * np.cos(t) + center[2]
                vertices.append([x, y, z])
        
        # Generate faces
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Create two triangles for each grid cell
                p1 = i * resolution + j
                p2 = p1 + 1
                p3 = p1 + resolution
                p4 = p3 + 1
                faces.append([p1, p2, p4])
                faces.append([p1, p4, p3])
        
        return np.array(vertices), np.array(faces)
    


    @staticmethod
    def generate_cylinder_mesh(start_point, end_point, radius, resolution=20):
        """
        Generates vertices and faces to approximate a cylinder between two points (start_point and end_point).
        
        Parameters:
        - start_point: A tuple or list with the (x, y, z) coordinates of the starting point.
        - end_point: A tuple or list with the (x, y, z) coordinates of the ending point.
        - radius: The radius of the cylinder.
        - resolution: The number of subdivisions around the circumference.
        
        Returns:
        - vertices: A list of vertices (each a list of [x, y, z]).
        - faces: A list of faces (each a list of 3 or 4 vertex indices).
        """
        # Vector from start to end
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        cylinder_vector = end_point - start_point
        
        # Length of the cylinder (height)
        height = np.linalg.norm(cylinder_vector)
        
        # Normalize cylinder_vector to get the direction
        direction = cylinder_vector / height
        
        # Generate the base circle in 2D (xy-plane) and then align the cylinder with the bond direction
        theta = np.linspace(0, 2 * np.pi, resolution)
        circle_base = np.array([[radius * np.cos(t), radius * np.sin(t), 0] for t in theta])

        # Use cross-product to find perpendicular vector for orientation
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        axis_length = np.linalg.norm(axis)
        
        if axis_length != 0:
            # Rotation matrix to align z-axis with the bond direction
            axis /= axis_length  # Normalize the axis
            angle = np.arccos(np.dot(z_axis, direction))
            rotation_matrix = MODrawer.create_rotation_matrix(axis, angle)
            circle_base = circle_base @ rotation_matrix.T  # Rotate the base circle
        else:
            # If bond is aligned with the z-axis, no rotation is needed
            rotation_matrix = np.eye(3)

        # Generate vertices for both ends of the cylinder
        vertices = []
        bottom_circle = circle_base + start_point  # Bottom circle vertices
        top_circle = circle_base + end_point  # Top circle vertices

        vertices.extend(bottom_circle)
        vertices.extend(top_circle)

        # Center points for bottom and top (used for the caps)
        vertices.append(start_point)  # Bottom center
        vertices.append(end_point)    # Top center

        # Create faces for the sides of the cylinder
        faces = []
        num_vertices = len(bottom_circle)
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([i, next_i, next_i + num_vertices])  # Side faces
            faces.append([i, next_i + num_vertices, i + num_vertices])  # Side faces

        # Create faces for the top and bottom caps
        bottom_center_index = 2 * num_vertices
        top_center_index = bottom_center_index + 1
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            faces.append([bottom_center_index, i, next_i])  # Bottom cap
            faces.append([top_center_index, i + num_vertices, next_i + num_vertices])  # Top cap

        return np.array(vertices), np.array(faces)

    @staticmethod
    def create_rotation_matrix(axis, angle):
        """
        Creates a rotation matrix to rotate points around a given axis by a given angle.
        
        Parameters:
        - axis: The axis to rotate around (must be a unit vector).
        - angle: The angle to rotate by (in radians).
        
        Returns:
        - A 3x3 rotation matrix.
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        x, y, z = axis

        return np.array([
            [cos_angle + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle, cos_angle + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
        ])



    def generate_orbital_mesh(self, key, orbital_scale=1):
        """
        Generate vertices, faces, and colors for a given molecular orbital.
        
        Parameters:
        - key: The key for the molecular orbital (e.g., 'MO1').
        - orbital_scale: Scale for the orbital lobes.
        
        Returns:
        - vertices: A list of all vertices.
        - faces: A list of all faces.
        - colors: A list of colors corresponding to the faces or vertices.
        """
        basis = self.mos.get(key)
        atom_centres = np.array(basis[:, :-1])
        atom_coefs = np.array(basis[:, -1])

        all_vertices = []
        all_faces = []
        all_colors = []  # List to hold colors for each vertex or face
        vertex_offset = 0

        # For the bonds
        all_bond_vertices = []
        all_bond_faces = []
        all_bond_colors = []  # List to hold colors for bonds
        bond_vertex_offset = 0
        
        for centre, atom_coef in zip(atom_centres, atom_coefs):
            # Generate sphere vertices and faces for each atomic centre
            radius = orbital_scale * np.abs(atom_coef) *3
            vertices, faces = self.generate_sphere_mesh(centre, radius, resolution=20)

            # Adjust face indices to account for the global vertex array
            faces += vertex_offset

            # Append vertices and faces to global lists
            all_vertices.extend(vertices)
            all_faces.extend(faces)

            # Normalize atom coefficient to range [0, 1] for colormap
            norm_coef = (atom_coef + 1) / 2  # Shift to range [0, 1]
            # Assign colors based on atom coefficient (like your original color mapping)
            if atom_coef >= 0:
                # color = (139, 0, 0)  # red
                color = warm_cmap(norm_coef)
            else:
                # color = (0, 139, 139)  # blue
                color = cool_cmap(norm_coef)
            rgb_color = tuple([int(255 * c) for c in color[:3]])
            # Add color to each face/vertex (depending on how you're rendering)
            for _ in faces:
                # all_colors.append(color)
                all_colors.append(rgb_color)

            # Update offset for the next set of faces
            vertex_offset += len(vertices)
        # Generate bond mesh
        list_coords = np.asarray(list(self.coordinates.values()))
        atom_types = np.asarray(list(self.coordinates.keys()))

        # Generate bond pairs and filter based on distance
        dist_pairs = np.asarray(list(combinations(list_coords, 2)))
        atom_pairs = np.asarray(list(combinations(atom_types, 2)))
        bool_array = np.linalg.norm(dist_pairs[:, 1] - dist_pairs[:, 0], axis=1) < 3.35  # Tolerance
        bond_array = dist_pairs[bool_array]
        atom_array = atom_pairs[bool_array]
        print(bond_array)
        print(atom_array)
        # for bond, atoms in zip(bond_array, atom_array):
        #     start_point, end_point = bond

        #     # Generate cylinder mesh for each bond
        #     bond_vertices, bond_faces = self.generate_cylinder_mesh(start_point, end_point, radius=0.05, resolution = 10)
            
        #     # Adjust bond face indices to account for the bond vertex array
        #     bond_faces += bond_vertex_offset

        #     # Append bond vertices and faces to global bond lists
        #     all_bond_vertices.extend(bond_vertices)
        #     all_bond_faces.extend(bond_faces)

        #     # Assign colors based on atom types in the bond
        #     atom1, atom2 = atoms
            
        #     # Assign RGB colors based on atom types in the bond
        #     if atom1.startswith('C') and atom2.startswith('Cl'):
        #         bond_color = ((0, 0, 0), (0, 255, 0))  # black, green
                
        #     elif atom1.startswith('Cl') and atom2.startswith('C'):
        #         bond_color = ((0, 255, 0), (0, 0, 0))  # green, black
        #     elif atom1.startswith('C') and atom2.startswith('N'):
        #         bond_color = ((0, 0, 0), (0, 0, 255))  # black, blue
        #     elif atom1.startswith('N') and atom2.startswith('C'):
        #         bond_color = ((0, 0, 255), (0, 0, 0))  # blue, black
        #     elif atom1.startswith('C') and atom2.startswith('C'):
        #         bond_color = ((0, 0, 0), (0, 0, 0))    # black, black
        #     elif atom1.startswith('Cl') and atom2.startswith('Cl'):
        #         bond_color = ((0, 255, 0), (0, 255, 0)) # green, green
        #     elif atom1.startswith('N') and atom2.startswith('N'):
        #         bond_color = ((0, 0, 255), (0, 0, 255)) # blue, blue
                
        #     else:
        #         bond_color = ((0, 0, 0), (0, 0, 0))    # Default black


        #     # Add bond colors for the two halves
        #     for i in range(len(bond_faces) // 2):  # Assuming 2 triangles for each half
        #         all_bond_colors.append(bond_color[0])  # First half color
        #     for i in range(len(bond_faces) // 2, len(bond_faces)):
        #         all_bond_colors.append(bond_color[1])  # Second half color

        #     # Update bond vertex offset for the next bond
        #     bond_vertex_offset += len(bond_vertices)
        # print(all_bond_colors)
        # print(all_bond_vertices)
        for bond, atoms in zip(bond_array, atom_array):
            start_point, end_point = bond

            # Calculate the midpoint of the bond
            mid_point = (start_point + end_point) / 2

            # Generate cylinder mesh for the first half (start -> midpoint)
            bond_vertices_1, bond_faces_1 = self.generate_cylinder_mesh(start_point, mid_point, radius=0.05, resolution=10)

            # Generate cylinder mesh for the second half (midpoint -> end)
            bond_vertices_2, bond_faces_2 = self.generate_cylinder_mesh(mid_point, end_point, radius=0.05, resolution=10)

            # Adjust face indices for both halves
            bond_faces_1 += bond_vertex_offset
            bond_faces_2 += bond_vertex_offset + len(bond_vertices_1)

            # Append vertices and faces to global lists for both halves
            all_bond_vertices.extend(bond_vertices_1)
            all_bond_vertices.extend(bond_vertices_2)
            all_bond_faces.extend(bond_faces_1)
            all_bond_faces.extend(bond_faces_2)

            # Assign RGB colors based on atom types in the bond
            atom1, atom2 = atoms
            if atom1.startswith('C') and atom2.startswith('Cl'):
                bond_color_1 = (0, 0, 0)  # black for first half
                bond_color_2 = (0, 98, 0)  # green for second half
            elif atom1.startswith('Cl') and atom2.startswith('C'):
                bond_color_1 = (0, 98, 0)  # green for first half
                bond_color_2 = (0, 0, 0)  # black for second half
            elif atom1.startswith('C') and atom2.startswith('N'):
                bond_color_1 = (0, 0, 0)  # black for first half
                bond_color_2 = (0, 0, 255)  # blue for second half
            elif atom1.startswith('N') and atom2.startswith('C'):
                bond_color_1 = (0, 0, 255)  # blue for first half
                bond_color_2 = (0, 0, 0)  # black for second half
            elif atom1.startswith('C') and atom2.startswith('C'):
                bond_color_1 = bond_color_2 = (0, 0, 0)  # both halves black
            elif atom1.startswith('Cl') and atom2.startswith('Cl'):
                bond_color_1 = bond_color_2 = (0, 98, 0)  # both halves green
            elif atom1.startswith('N') and atom2.startswith('N'):
                bond_color_1 = bond_color_2 = (0, 0, 255)  # both halves blue
            else:
                bond_color_1 = bond_color_2 = (0, 0, 0)  # Default black

            # Add bond colors for each half
            for _ in bond_faces_1:
                all_bond_colors.append(bond_color_1)  # Color for the first half
            for _ in bond_faces_2:
                all_bond_colors.append(bond_color_2)  # Color for the second half

            # Update bond vertex offset for the next bond
            bond_vertex_offset += len(bond_vertices_1) + len(bond_vertices_2)
        # print(all_vertices[:2])
        return (
            np.array(all_vertices), np.array(all_faces), all_colors,
            np.array(all_bond_vertices), np.array(all_bond_faces), all_bond_colors
        )

    def get_center(self):
        coordinates = np.array(list(self.coordinates.values()))
        x_mid = np.sum(coordinates[:,0]) / len(coordinates)
        y_mid = np.sum(coordinates[:,1]) / len(coordinates)
        z_mid = np.sum(coordinates[:,2]) / len(coordinates)

        return np.array([x_mid, y_mid, z_mid])

 




if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'New folder/ttm_id3.out')

    ttm = MolecularOrbitals(path, 'ttm')
    print(ttm.coordinates)
    coords_array = np.array(list(ttm.coordinates.values()))
    x_mid = np.sum(coords_array[:,0]) / len(coords_array)
    y_mid = np.sum(coords_array[:,1]) / len(coords_array)
    z_mid = np.sum(coords_array[:,2]) / len(coords_array)
    print(coords_array)
    print(x_mid,y_mid,z_mid)




