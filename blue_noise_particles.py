import bpy
import math
from mathutils.kdtree import KDTree
from functools import total_ordering
import heapq

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


@total_ordering
class HeapItem:
    def __init__(self, weight, index):
        self.weight = weight
        self.index = index

    def __lt__(self, other):
        return self.weight > other.weight

class SampleEliminator:
    def __init__(self, locations, target_samples):
        self.locations = locations
        self.tree = KDTree(len(locations))
        for index, location in locations.items():
            self.tree.insert(location, index)
        self.tree.balance()
        self.rmax = 0.1  # TODO: Choose this appropriately
        self.alpha = 8
        self.target_samples = target_samples
        self.current_samples = len(self.locations)

        gamma = 1.5
        beta = 0.65
        M = self.current_samples
        N = self.target_samples
        self.rmin = self.rmax * (1 - (N / M) ** gamma) * beta

        # Build initial heap
        self.heap = []
        self.heap_items = {}
        for index, location in locations.items():
            tot_weight = 0
            for location2, index2, d in self.tree.find_range(location, 2 * self.rmax):
                tot_weight += self.w(d)
            item = HeapItem(tot_weight, index)
            self.heap_items[index] = item
            heapq.heappush(self.heap, item)

    def eliminate_one(self):
        item = heapq.heappop(self.heap)
        index = item.index
        print("eliminating", index)
        location = self.locations[index]
        for location2, index2, d in self.tree.find_range(location, 2 * self.rmax):
            item2 = self.heap_items[index2]
            item2.weight -= self.w(d)
        heapq.heapify(self.heap)
        self.current_samples -= 1

    def eliminate(self):
        print("elimiate")
        print(self.current_samples)
        print(self.target_samples)
        while self.current_samples > self.target_samples:
            self.eliminate_one()

    def get_indices(self):
        return (item.index for item in self.heap)

    def d(self, i, j):
        li = self.locations[i]
        lj = self.locations[j]
        return math.sqrt((li[0] - lj[0]) ** 2 +
                         (li[1] - lj[1]) ** 2 +
                         (li[2] - lj[2]) ** 2)

    def adj_d(self, d):
        return min(d, 2 * self.rmax)

    def w(self, d):
        return (1 - self.adj_d(d) / 2 / self.rmax) ** self.alpha




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
        #pset.count = 10000
        pset.show_unborn = True

        particles = psys.particles

        locations = dict((index, particle.location) for (index, particle) in particles.items())
        se = SampleEliminator(locations, 500)
        se.eliminate()

        alive_indices = set(se.get_indices())
        print(alive_indices)

        for index, particle in particles.items():
            alive = index in alive_indices
            particle.alive_state = 'ALIVE' if alive else 'DEAD'
            if not alive:
                particle.location = [0,0,0]

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
