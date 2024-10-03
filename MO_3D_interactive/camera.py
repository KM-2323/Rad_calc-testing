# import pygame as pg
# from matrix_functions import *

# class Camera:
#     def __init__(self, render, position):
#         self.render = render
#         self.position = np.array([*position, 1.0])
#         self.forward = np.array([0, 0, 1, 1])
#         self.up = np.array([0, 1, 0, 1])
#         self.right = np.array([1, 0, 0, 1])
#         self.h_fov = math.pi / 3
#         self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)
#         self.near_plane = 0.1
#         self.far_plane = 100
#         self.moving_speed = 0.3
#         self.rotation_speed = 0.015

#         self.anglePitch = 0
#         self.angleYaw = 0
#         self.angleRoll = 0

#     def control(self):
#         key = pg.key.get_pressed()
#         if key[pg.K_a]:
#             self.position -= self.right * self.moving_speed
#         if key[pg.K_d]:
#             self.position += self.right * self.moving_speed
#         if key[pg.K_w]:
#             self.position += self.forward * self.moving_speed
#         if key[pg.K_s]:
#             self.position -= self.forward * self.moving_speed
#         if key[pg.K_q]:
#             self.position += self.up * self.moving_speed
#         if key[pg.K_e]:
#             self.position -= self.up * self.moving_speed

#         if key[pg.K_LEFT]:
#             self.camera_yaw(-self.rotation_speed)
#         if key[pg.K_RIGHT]:
#             self.camera_yaw(self.rotation_speed)
#         if key[pg.K_UP]:
#             self.camera_pitch(-self.rotation_speed)
#         if key[pg.K_DOWN]:
#             self.camera_pitch(self.rotation_speed)

#     def camera_yaw(self, angle):
#         self.angleYaw += angle

#     def camera_pitch(self, angle):
#         self.anglePitch += angle

#     def axiiIdentity(self):
#         self.forward = np.array([0, 0, 1, 1])
#         self.up = np.array([0, 1, 0, 1])
#         self.right = np.array([1, 0, 0, 1])

#     def camera_update_axii(self):
#         # rotate = rotate_y(self.angleYaw) @ rotate_x(self.anglePitch)
#         rotate = rotate_x(self.anglePitch) @ rotate_y(self.angleYaw)  # this concatenation gives right visual
#         self.axiiIdentity()
#         self.forward = self.forward @ rotate
#         self.right = self.right @ rotate
#         self.up = self.up @ rotate

#     def camera_matrix(self):
#         self.camera_update_axii()
#         return self.translate_matrix() @ self.rotate_matrix()

#     def translate_matrix(self):
#         x, y, z, w = self.position
#         return np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [-x, -y, -z, 1]
#         ])

#     def rotate_matrix(self):
#         rx, ry, rz, w = self.right
#         fx, fy, fz, w = self.forward
#         ux, uy, uz, w = self.up
#         return np.array([
#             [rx, ux, fx, 0],
#             [ry, uy, fy, 0],
#             [rz, uz, fz, 0],
#             [0, 0, 0, 1]
#         ])
    
#     def project_point(self, point):
#         # Convert the 3D point into homogeneous coordinates (add 1 as the fourth coordinate)
#         point_3d = np.array([*point, 1])

#         # Apply camera matrix and projection matrix
#         transformed_point = point_3d @ self.camera_matrix() @ self.render.projection.projection_matrix

#         # Normalize by the fourth (homogeneous) coordinate
#         transformed_point /= transformed_point[-1]

#         # Apply screen transformation to get 2D coordinates
#         screen_point = transformed_point @ self.render.projection.to_screen_matrix

#         # Return the 2D screen coordinates (ignoring the z coordinate)
#         return screen_point[:2]






# import pygame as pg
# from matrix_functions import *  # Assuming your matrix operations are defined here
# import numpy as np
# import math

# class Camera:
#     def __init__(self, render, position):
#         self.render = render
#         self.position = np.array([*position, 1.0])  # Camera's position in 3D space
#         self.forward = np.array([0, 0, 1, 1])      # Forward direction vector
#         self.up = np.array([0, 1, 0, 1])           # Up direction vector
#         self.right = np.array([1, 0, 0, 1])        # Right direction vector
#         self.h_fov = math.pi / 3                   # Horizontal field of view
#         self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)  # Vertical FOV
#         self.near_plane = 0.1
#         self.far_plane = 100
#         self.moving_speed = 0.3                    # Speed of movement (translation)
#         self.rotation_speed = 0.015                # Speed of rotation

#         self.anglePitch = 0                        # Camera pitch (up/down rotation)
#         self.angleYaw = 0                          # Camera yaw (left/right rotation)

#     def control(self):
#         """Handle key inputs for camera movement, rotation, and zoom."""
#         key = pg.key.get_pressed()

#         # Translation (WASD movement)
#         if key[pg.K_w]:  # Move forward
#             self.position += self.forward * self.moving_speed
#         if key[pg.K_s]:  # Move backward
#             self.position -= self.forward * self.moving_speed
#         if key[pg.K_a]:  # Move left
#             self.position -= self.right * self.moving_speed
#         if key[pg.K_d]:  # Move right
#             self.position += self.right * self.moving_speed

#         # Zoom in/out (Z to zoom in, C to zoom out)
#         if key[pg.K_z]:  # Zoom in (move forward)
#             self.position += self.forward * self.moving_speed * 2  # Faster zoom
#         if key[pg.K_c]:  # Zoom out (move backward)
#             self.position -= self.forward * self.moving_speed * 2

#         # Rotation (Q/E for yaw rotation)
#         if key[pg.K_q]:  # Rotate left (yaw)
#             self.camera_yaw(-self.rotation_speed)
#         if key[pg.K_e]:  # Rotate right (yaw)
#             self.camera_yaw(self.rotation_speed)

#     def camera_yaw(self, angle):
#         """Rotate the camera left/right (yaw) by a given angle."""
#         self.angleYaw += angle

#     def camera_pitch(self, angle):
#         """Rotate the camera up/down (pitch) by a given angle."""
#         self.anglePitch += angle

#     def axiiIdentity(self):
#         """Reset the camera's axes to the default orientation."""
#         self.forward = np.array([0, 0, 1, 1])
#         self.up = np.array([0, 1, 0, 1])
#         self.right = np.array([1, 0, 0, 1])

#     def camera_update_axii(self):
#         """Update the camera's direction vectors (forward, right, up) based on pitch and yaw."""
#         # Apply rotation matrices for pitch (up/down) and yaw (left/right)
#         rotate = rotate_x(self.anglePitch) @ rotate_y(self.angleYaw)
#         self.axiiIdentity()  # Reset the axes before applying rotations
#         self.forward = self.forward @ rotate
#         self.right = self.right @ rotate
#         self.up = self.up @ rotate

#     def camera_matrix(self):
#         """Return the combined translation and rotation matrix for the camera."""
#         self.camera_update_axii()  # Update direction vectors
#         return self.translate_matrix() @ self.rotate_matrix()

#     def translate_matrix(self):
#         """Return the translation matrix based on the camera's position."""
#         x, y, z, w = self.position
#         return np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [-x, -y, -z, 1]  # Translate by the inverse of the camera's position
#         ])

#     def rotate_matrix(self):
#         """Return the rotation matrix based on the camera's current orientation (forward, up, right)."""
#         rx, ry, rz, w = self.right
#         fx, fy, fz, w = self.forward
#         ux, uy, uz, w = self.up
#         return np.array([
#             [rx, ux, fx, 0],
#             [ry, uy, fy, 0],
#             [rz, uz, fz, 0],
#             [0, 0, 0, 1]
#         ])
    
#     def project_point(self, point):
#         """Project a 3D point into 2D screen space."""
#         # Convert the 3D point into homogeneous coordinates (add 1 as the fourth coordinate)
#         point_3d = np.array([*point, 1])

#         # Apply camera matrix and projection matrix
#         transformed_point = point_3d @ self.camera_matrix() @ self.render.projection.projection_matrix

#         # Normalize by the fourth (homogeneous) coordinate (perspective divide)
#         transformed_point /= transformed_point[-1]

#         # Apply screen transformation to get 2D coordinates
#         screen_point = transformed_point @ self.render.projection.to_screen_matrix

#         # Return the 2D screen coordinates (ignoring the z coordinate)
#         return screen_point[:2]
import pygame as pg
from matrix_functions import *
import numpy as np
import math

class Camera:
    def __init__(self, render, position):
        self.render = render
        self.position = np.array([*position, 1.0])  # Camera's position in 3D space
        self.forward = np.array([0, 0, 1, 1])      # Forward direction vector
        self.up = np.array([0, 1, 0, 1])           # Up direction vector
        self.right = np.array([1, 0, 0, 1])        # Right direction vector
        self.h_fov = math.pi / 3                   # Horizontal field of view
        self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)  # Vertical FOV
        self.near_plane = 0.1
        self.far_plane = 100
        self.moving_speed = 0.3                    # Speed of movement (translation)
        self.rotation_speed = 0.015                # Speed of rotation

        self.angleYaw = 0                          # Camera yaw (left/right rotation)
        self.anglePitch = 0                        # Camera pitch (up/down rotation)

    def control(self):
        """Handle key inputs for camera movement, rotation, and zoom."""
        key = pg.key.get_pressed()

        # Translation (WASD movement)
        if key[pg.K_w]:  # Move forward
            self.position -= self.forward * self.moving_speed
        if key[pg.K_s]:  # Move backward
            self.position += self.forward * self.moving_speed
        if key[pg.K_a]:  # Move left
            self.position += self.right * self.moving_speed
        if key[pg.K_d]:  # Move right
            self.position -= self.right * self.moving_speed

        # Camera Yaw rotation around the view (rotate around up vector)
        if key[pg.K_z]:
            self.render.object.rotate_molecule_x(self.rotation_speed)  # Rotate molecule around X-axis
        if key[pg.K_c]:
            self.render.object.rotate_molecule_z(self.rotation_speed)  # Rotate molecule around Z-axis

        # Rotation (Q/E for yaw rotation)
        if key[pg.K_q]:  # Rotate left (yaw)
            self.camera_yaw(-self.rotation_speed)
        if key[pg.K_e]:  # Rotate right (yaw)
            self.camera_yaw(self.rotation_speed)

        # Optionally, add vertical movement using up and down arrow keys
        if key[pg.K_UP]:
            self.camera_pitch(-self.rotation_speed)
        if key[pg.K_DOWN]:
            self.camera_pitch(self.rotation_speed)

    def camera_yaw(self, angle):
        """Rotate the camera left/right (yaw) by a given angle."""
        self.angleYaw += angle

    def camera_pitch(self, angle):
        """Rotate the camera up/down (pitch) by a given angle."""
        self.anglePitch += angle

    def axiiIdentity(self):
        """Reset the camera's axes to the default orientation."""
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

    def camera_update_axii(self):
        """Update the camera's direction vectors (forward, right, up) based on pitch and yaw."""
        # Apply rotation matrices for pitch (up/down) and yaw (left/right)
        rotate = rotate_x(self.anglePitch) @ rotate_y(self.angleYaw)
        self.axiiIdentity()  # Reset the axes before applying rotations
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

    def camera_matrix(self):
        """Return the combined translation and rotation matrix for the camera."""
        self.camera_update_axii()  # Update direction vectors
        return self.translate_matrix() @ self.rotate_matrix()
    
    def rotate_on_view(self, angle):
        """Rotate the camera around its current up vector (yaw rotation)"""
        # Create a rotation matrix around the camera's up vector
        rotation_matrix = rotate_y(angle)  # Rotate around the local Y (up) axis

        # Update the forward and right vectors based on the rotation
        self.forward = self.forward @ rotation_matrix
        self.right = self.right @ rotation_matrix


    


    def translate_matrix(self):
        """Return the translation matrix based on the camera's position."""
        x, y, z, w = self.position
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-x, -y, -z, 1]  # Translate by the inverse of the camera's position
        ])

    def rotate_matrix(self):
        """Return the rotation matrix based on the camera's current orientation (forward, up, right)."""
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ])
    
    def project_point(self, point):
        """Project a 3D point into 2D screen space."""
        # Convert the 3D point into homogeneous coordinates (add 1 as the fourth coordinate)
        point_3d = np.array([*point, 1])

        # Apply camera matrix and projection matrix
        transformed_point = point_3d @ self.camera_matrix() @ self.render.projection.projection_matrix

        # Normalize by the fourth (homogeneous) coordinate (perspective divide)
        transformed_point /= transformed_point[-1]

        # Apply screen transformation to get 2D coordinates
        screen_point = transformed_point @ self.render.projection.to_screen_matrix

        # Return the 2D screen coordinates (ignoring the z coordinate)
        return screen_point[:2]
