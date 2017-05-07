bl_info = {
    "name": "Blue Noise Particles",
    "description": "",
    "author": "Adam Newgas",
    "version": (0, 0, 1),
    "blender": (2, 78, 0),
    "location": "",
    "warning": "",
    "wiki_url": "",
    "category": ""}

import bpy
import bmesh
from mathutils.kdtree import KDTree


class BlueNoiseParticles(bpy.types.Operator):
    bl_idname = "object.blue_noise_particles_operator"
    bl_label = "Blue Noise Particles"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ((ob is not None) and
                (ob.mode == "OBJECT") and
                (ob.type == "MESH") and
                (context.mode == "OBJECT"))

    def execute(self, context):
        # Create the new object
        obj = context.active_object  # type: bpy.types.Object

        bpy.ops.object.particle_system_add()
        psys = obj.particle_systems[-1]  # type: bpy.types.ParticleSystem
        pset = psys.settings
        pset.count = 2000
        pset.show_unborn = True

        particles = psys.particles

        tree = KDTree(len(particles))

        for index, particle in particles.items():
            tree.insert(particle.location, index)

        tree.balance()

        co, index, dist = tree.find([0,0,0])
        print(co)

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.operator(BlueNoiseParticles.bl_idname,
                         text="Blue Noise Particles",
                         icon='PLUGIN')

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_curve_add.append(menu_func)


def unregister():
    bpy.types.INFO_MT_curve_add.remove(menu_func)
    bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
    register()
