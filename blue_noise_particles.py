# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import math
import fibonacci_heap_mod
import bpy
import bpy.props
import bpy.utils
import bmesh
import mathutils
import mathutils.geometry
import mathutils.kdtree

bl_info = {
    "name": "Blue Noise Particles",
    "description": "",
    "author": "Adam Newgas",
    "version": (0, 0, 1),
    "blender": (2, 78, 0),
    "location": "View3D > Add > Mesh > Blue Noise Particles",
    "warning": "",
    "wiki_url": "https://github.com/BorisTheBrave/blue-noise-particles/wiki",
    "category": "Add Mesh"}


class SampleEliminator:
    def __init__(self, locations, densities, target_samples, is_volume, mesh_area):
        self.locations = locations

        # Setup a KD Tree of all lcations
        self.tree = mathutils.kdtree.KDTree(len(locations))
        for index, location in enumerate(locations):
            self.tree.insert(location, index)
        self.tree.balance()

        self.alpha = 8
        self.target_samples = target_samples
        self.current_samples = len(self.locations)

        M = self.current_samples
        N = self.target_samples

        # Choose rmax via heuristic
        bounds = [max(p[i] for p in locations) - min(p[i] for p in locations)
                  for i in range(3)]

        # Volume based constraint
        A = bounds[0] * bounds[1] * bounds[2]
        rmax3 = (A / 4 / math.sqrt(2) / N) ** (1 / 3)

        # Volume estimate only valid for reasonably squarish things
        is_thin = rmax3 <= min(bounds)
        if is_thin:
            rmax3 = float('inf')

        if is_thin or not is_volume:
            # If we are constrained to 2d surface, then it is possible to
            # get a better bound for rmax. Depends on the mesh geometry.
            rmax2 = math.sqrt(mesh_area / 2 / math.sqrt(3) / N)
        else:
            rmax2 = float('inf')

        self.rmax = min(rmax2, rmax3)

        if densities is not None:
            # Need to be a bit more conservative if the faces are imbalanced.
            # This could still go wrong with extreme vertex weights...
            self.rmax *= 3
            dmax = max(densities)
            self.densities = [d / dmax for d in densities]
        else:
            self.densities = [1] * len(locations)

        # Choose rmin via heuristic
        gamma = 1.5
        beta = 0.65
        self.rmin = self.rmax * (1 - (N / M) ** gamma) * beta

        # Build initial heap
        self.heap = fibonacci_heap_mod.Fibonacci_heap()
        self.heap_items = {}
        for index, location in enumerate(locations):
            tot_weight = 0
            for location2, index2, d in self.tree.find_range(location, 2 * self.rmax):
                tot_weight += self.w(d, index, index2)
            item = self.heap.enqueue(index, -tot_weight)
            self.heap_items[index] = item

    def eliminate_one(self):
        # Extract highest priority item
        item = self.heap.dequeue_min()
        index = item.get_value()
        del self.heap_items[index]
        location = self.locations[index]

        # Update all adjacent items
        for location2, index2, d in self.tree.find_range(location, 2 * self.rmax):
            item2 = self.heap_items.get(index2)
            if item2:
                new_weight = item2.get_priority() + self.w(d, index2, index)
                self.heap.delete(item2)
                item2 = self.heap.enqueue(index2, new_weight)
                self.heap_items[index2] = item2
        self.current_samples -= 1

    def eliminate(self):
        while self.current_samples > self.target_samples:
            self.eliminate_one()

    def get_indices(self):
        return list(self.heap_items.keys())

    def d(self, i, j):
        li = self.locations[i]
        lj = self.locations[j]
        return math.sqrt((li[0] - lj[0]) ** 2 +
                         (li[1] - lj[1]) ** 2 +
                         (li[2] - lj[2]) ** 2)

    def w(self, d, i, j):
        # This sqrt is important as it ensures our distance scale (a length)
        # is consistent with the density scale (an area^-1)
        # If they are not consistent, then the distribution that generated
        # the points won't be close to what we are eliminating down to
        # leading to poor quality results.
        d *= math.sqrt(self.densities[i])
        adj_d = min(d, 2 * self.rmax)
        return (1 - adj_d / 2 / self.rmax) ** self.alpha


def get_mesh_area(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    area = sum(f.calc_area() for f in bm.faces)
    bm.free()
    return area

def particle_distribute(obj, particle_count, emit_from, scene):
    # Uses blender's built in particle system.
    # Sadly doesn't work with density weighting as it is not possible
    # to extract vertex densities
    bpy.ops.object.particle_system_add()
    psys = obj.particle_systems[-1]  # type: bpy.types.ParticleSystem
    pset = psys.settings
    pset.count = particle_count
    pset.emit_from = emit_from
    pset.distribution = 'RAND'
    pset.use_even_distribution = True

    # Force a scene update (generates particle loations)
    scene.update()

    # Extract locations
    particles = psys.particles
    locations = [particle.location for (index, particle) in particles.items()]

    # Delete particle system
    bpy.ops.object.particle_system_remove()

    return locations, None

V1 = mathutils.Vector([0, 0, 0])
V2 = mathutils.Vector([0, 0, 1])
V3 = mathutils.Vector([0, 1, 0])


def weighted_particle_distribute(obj, particle_count, weight_group):
    # Distributes points similarly to blender's built in particle system
    # for emit_from=FACES, random type=RAND, even_distribution=True
    # and with the given vertex weight group for density.
    # It returns the locations of the particles, and the density of the
    # area that particle was found
    import numpy as np
    np.random.seed(0)

    # This is a rough port of what the C code does for random particles
    # See particle_distribute.c, distribute_from_faces_exec

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # TODO: The original code handles quads.
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.faces.ensure_lookup_table()
    areas = np.array([f.calc_area() for f in bm.faces])

    # Compute face relative densities
    group_index = obj.vertex_groups[weight_group].index
    layer = bm.verts.layers.deform[0]
    face_densities = np.zeros(len(bm.faces))
    for face in bm.faces:
        w = np.mean([v[layer].get(group_index, 0) for v in face.verts])
        face_densities[face.index] = w
    print(face_densities)
    print(areas)
    areas *= face_densities


    # Precompute distribution amongst faces
    careas = areas.cumsum()
    total_area = careas[-1]

    # Randomly pick which face each particle goes to
    rand_index = np.random.uniform(0, total_area, particle_count)
    face_indices = np.searchsorted(careas, rand_index)

    # Randomly pick where in each face the particle is
    rand_u = np.random.uniform(size=particle_count)
    rand_v = np.random.uniform(size=particle_count)
    rand_sum = rand_u + rand_v

    V = mathutils.Vector
    locations = []
    densities = []
    for i in range(particle_count):
        face_index = face_indices[i]
        face = bm.faces[face_index]
        vc = len(face.verts)
        assert vc == 3
        u = rand_u[i]
        v = rand_v[i]
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        loc = mathutils.geometry.barycentric_transform(
            V([0, u, v]),
            V1,
            V2,
            V3,
            face.verts[0].co,
            face.verts[1].co,
            face.verts[2].co,
        )
        loc = obj.matrix_world * loc
        locations.append(loc)
        densities.append(face_densities[face_index])


    bm.free()
    return locations, densities

class BlueNoiseParticles(bpy.types.Operator):
    bl_idname = "object.blue_noise_particles_operator"
    bl_label = "Blue Noise Particles"
    bl_options = {'REGISTER', 'UNDO'}

    emit_from_types = [("VERT", "Verts", "Emit from vertices"),
                       ("FACE", "Faces", "Emit from faces"),
                       ("VOLUME", "Volume", "Emit from volume")]
    emit_from = bpy.props.EnumProperty(items=emit_from_types,
                                       name="Emit From",
                                       description="Controls where particles are generated",
                                       default="FACE")

    quality_types = [("1", "None", ""),
                     ("1.5", "Low", ""),
                     ("2", "Medium", ""),
                     ("5", "High", "")]
    quality = bpy.props.EnumProperty(items=quality_types,
                                     name="Quality",
                                     description="Controls how much oversampling is done",
                                     default="2")

    count = bpy.props.IntProperty(name="Count",
                                  description="Number of particles to emit",
                                  default=1000,
                                  min=0)

    vertex_group_density = bpy.props.StringProperty(name="Density",
                                      description="Vertex group to control density")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ((ob is not None) and
                (ob.mode == "OBJECT") and
                (ob.type == "MESH") and
                (context.mode == "OBJECT"))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "emit_from")
        layout.prop(self, "quality")
        layout.prop(self, "count")
        obj = bpy.data.objects[self.obj_name]
        layout.prop_search(self, "vertex_group_density", obj, "vertex_groups", text="Density")

    def execute(self, context):
        obj = context.active_object  # type: bpy.types.Object
        scene = context.scene

        initial_particle_count = int(self.count * float(self.quality))

        is_volume = self.emit_from == 'VOLUME'
        mesh_area = None
        if not is_volume:
            mesh_area = get_mesh_area(obj)

        if not self.vertex_group_density:
            locations, densities = particle_distribute(obj, initial_particle_count, self.emit_from, scene)
        else:
            if self.emit_from != "FACE":
                self.report("Vertex group density only supported when emitting from faces", 'ERROR_INVALID_INPUT')
                return {"FINISHED"}
            locations, densities = weighted_particle_distribute(obj, initial_particle_count, self.vertex_group_density)

        # Run sample elimination
        se = SampleEliminator(locations, densities, self.count, is_volume, mesh_area)
        se.eliminate()
        alive_indices = se.get_indices()
        alive_locations = [tuple(locations[i]) for i in alive_indices]


        # Create a new object, with vertices according the the alive locations
        me = bpy.data.meshes.new(obj.name + " ParticleMesh")
        ob = bpy.data.objects.new(obj.name + " Particles", me)
        me.from_pydata(alive_locations, [], [])
        scene.objects.link(ob)
        me.update()

        # Select new object
        scene.objects.active = ob
        obj.select = False
        ob.select = True

        # Add a particle system to the new object
        bpy.ops.object.particle_system_add()
        psys = ob.particle_systems[-1]  # type: bpy.types.ParticleSystem
        pset = psys.settings
        pset.count = self.count
        pset.emit_from = 'VERT'
        pset.use_emit_random = False
        pset.frame_start = 0
        pset.frame_end = 0
        pset.use_render_emitter = False
        pset.physics_type = 'NO'

        return {'FINISHED'}

    def invoke(self, context, event):
        self.obj_name = context.active_object.name
        return context.window_manager.invoke_props_dialog(self)

def menu_func(self, context):
    self.layout.operator(BlueNoiseParticles.bl_idname,
                         text="Blue Noise Particles",
                         icon='PLUGIN')


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_mesh_add.append(menu_func)


def unregister():
    bpy.types.INFO_MT_mesh_add.remove(menu_func)
    bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
    register()
