Blue Noise Particles
=====================

This Blender plugin generates a random arrangement of particles with a blue noise distribution. 
This is also known as Poisson Disk Sampling.

This distribution of particles guarantees no two particles are very near each other. It's often considered a higher 
quality particle arrangement than Blender's default uniform sampling. It's particularly useful for organic 
arrangements, and randomly arranging meshes without collisions.

The particular method of noise generation is called [Sample Elimination for Poisson Disk Sample Sets][1]. Thanks to
Cem Yuksel for the research and the clear statement of the method.
  
[1]: http://www.cemyuksel.com/research/sampleelimination/

Installation
------------
Download the zip, then go to `File > User Preferences > Addons`, and search for the Blue Noise Particles addon, 
enabling it. 

Or you can simply copy the python files from this repository into your Blender Addon dir, then enable the addon. 

Usage
-----

Select any mesh, then run the plugin from the `Add > Mesh` menu, or the toolbox. 
The parameters work extremely similarly to the normal particle options.

The plugin creates a new mesh with a particle system attached. You can then customize the particle system as
normal, for example changing the render type to Object to draw a mesh at the location of each particle.

Further explanation and examples can be found in the wiki on github: 
<https://github.com/boristhebrave/blue-noise-particles/wiki>

Further external reading can be found at:

* <http://devmag.org.za/2009/05/03/poisson-disk-sampling/>

* <http://www.cemyuksel.com/research/sampleelimination/sampleelimination.pdf>


License
-------
This code is licensed under the GPL License copyright Adam Newgas 2017.

fibonacci_heap_mod has been attached for convenience. It is available under the Apache v2 License.

