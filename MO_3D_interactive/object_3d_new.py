import pygame as pg
import numpy as np
from matrix_functions import *
from numba import njit


@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Object3D:
    def __init__(self, render, orbital_data, bond_data):
        self.render = render
        
        # Orbital mesh data
        orbital_vertices, orbital_faces, orbital_colors = orbital_data
        
        # Bond mesh data
        bond_vertices, bond_faces, bond_colors = bond_data

        # Store orbital data
        self.orbital_vertices = np.hstack([np.array(orbital_vertices), np.ones((len(orbital_vertices), 1))])
        self.orbital_faces = orbital_faces
        self.orbital_colors = orbital_colors

        # Store bond data
        self.bond_vertices = np.hstack([np.array(bond_vertices), np.ones((len(bond_vertices), 1))])
        self.bond_faces = bond_faces
        self.bond_colors = bond_colors
        
        self.movement_flag = False
        self.draw_vertices = False
        self.label = ''
        
        # Initial translation (to avoid 0 coordinates)
        self.translate([0.0001, 0.0001, 0.0001])
        
    def draw(self):
        
        self.screen_projection_bonds(self.bond_vertices, self.bond_faces, self.bond_colors)
        self.screen_projection(self.orbital_vertices, self.orbital_faces, self.orbital_colors)
        self.movement()

    def screen_projection(self, vertices, faces, colors):
        # Project vertices from 3D to 2D for rendering
        vertices = vertices @ self.render.camera.camera_matrix()
        vertices = vertices @ self.render.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.render.projection.to_screen_matrix
        vertices = vertices[:, :2]

        # Draw the projected faces
        for index, face in enumerate(faces):
            color = colors[index]
            polygon = vertices[face]
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                pg.draw.polygon(self.render.screen, color, polygon, 0)

        if self.draw_vertices:
            for vertex in vertices:
                if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
                    pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

    def screen_projection_bonds(self, vertices, faces, colors):
        # Project vertices from 3D to 2D for rendering (bonds)
        vertices = vertices @ self.render.camera.camera_matrix()
        vertices = vertices @ self.render.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.render.projection.to_screen_matrix
        vertices = vertices[:, :2]

        # Handle splitting bond faces into two parts for color assignment
        
        for index, face in enumerate(faces):
            color = colors[index]
            polygon = vertices[face]
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                # Draw the two halves of the bond with different colors
                # if index < half:
                #     pg.draw.polygon(self.render.screen, color, polygon, 0)  # First color
                # else:
                #     pg.draw.polygon(self.render.screen, color, polygon, 0)  # Second color
                pg.draw.polygon(self.render.screen, color, polygon, 0)

        # if self.draw_vertices:
        #     for vertex in vertices:
        #         if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
        #             pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005))

    # Rotation transformations for both orbital and bond vertices
    def rotate_molecule_x(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_x(angle)
        self.bond_vertices = self.bond_vertices @ rotate_x(angle)

    def rotate_molecule_y(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_y(angle)
        self.bond_vertices = self.bond_vertices @ rotate_y(angle)

    def rotate_molecule_z(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_z(angle)
        self.bond_vertices = self.bond_vertices @ rotate_z(angle)

    # Translation and scaling methods for both orbital and bond vertices
    def translate(self, pos):
        self.orbital_vertices = self.orbital_vertices @ translate(pos)
        self.bond_vertices = self.bond_vertices @ translate(pos)

    def scale(self, scale_to):
        self.orbital_vertices = self.orbital_vertices @ scale(scale_to)
        self.bond_vertices = self.bond_vertices @ scale(scale_to)

    def rotate_x(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_x(angle)
        self.bond_vertices = self.bond_vertices @ rotate_x(angle)

    def rotate_y(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_y(angle)
        self.bond_vertices = self.bond_vertices @ rotate_y(angle)

    def rotate_z(self, angle):
        self.orbital_vertices = self.orbital_vertices @ rotate_z(angle)
        self.bond_vertices = self.bond_vertices @ rotate_z(angle)
