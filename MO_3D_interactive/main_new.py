import pygame as pg
from object_3d_new import Object3D
from camera import Camera
from projection import Projection
from sphere_render_new import MODrawer
import os

class SoftwareRender:
    def __init__(self, mo_key, camera_view):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1600, 900
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()

        # Load molecular data
        self.molecular_data = MODrawer(output_file_path1, 'compound_name')

        # Generate orbital and bond meshes based on the selected molecular orbital
        self.create_objects(mo_key, camera_view)

    def create_objects(self, mo_key, camera_view):
        centroid = self.molecular_data.get_center()
        
        # Set up the camera position based on the view
        if camera_view == 'xy':
            camera_position = [0, 0, 50] + centroid
        elif camera_view == 'xz':
            camera_position = [0, 50, 0] + centroid
        elif camera_view == 'yz':
            camera_position = [50, 0, 0] + centroid
        elif camera_view == '45deg':
            camera_position = [70, 70, 70] + centroid
        else:
            camera_position = [10, 0, 100]  # Default

        # Initialize the camera and projection
        self.camera = Camera(self, camera_position)
        self.projection = Projection(self)

        # Get molecular orbital data (orbital and bond vertices, faces, and colors)
        orbital_data = self.molecular_data.generate_orbital_mesh(mo_key, orbital_scale=1)[:3]
        bond_data = self.molecular_data.generate_orbital_mesh(mo_key, orbital_scale=1)[3:]

        # Pass the orbital and bond data to the Object3D class
        self.object = Object3D(self, orbital_data=orbital_data, bond_data=bond_data)

    def draw(self):
        self.screen.fill(pg.Color('white'))
        self.object.draw()

    def run(self):
        while True:
            self.draw()
            self.camera.control()
            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick(self.FPS)


if __name__ == "__main__":
    mo_key = 'MO4'  # Replace this with dynamic user input if needed
    camera_view = 'xy'  # Default view
    current_directory = os.getcwd()
    output_file_path1 = os.path.join(current_directory, 'New folder/ttm.out')
    app = SoftwareRender(mo_key, camera_view)
    app.run()
