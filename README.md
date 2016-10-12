CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3** 

* Xiang Deng
* Tested on:  Windows 10-Home, i7-6700U @ 2.6GHz 16GB, GTX 1060 6GB (Personal Computer)

* Features
  * BSDF shading with diffusion, reflection.
  * Stream Compaction for removing inactive rays.
  * Sorting rays by material id.
  * First bounce caching of intersections.


* Additional features
  * Work efficient shared memory compaction.
  * Refractions.
  * Anti-aliasing with uniform guassian spherical jitter.
  * Motion blur.
  * Depth of field. 


Motion blur + basic shading features (reflection + diffusion + refraction) | Blur Screen Shot From Differnt Angle | Static scene rendering + basic shading features (reflection + diffusion + refraction)
:-------------------------:|:-------------------------:
 ![](img/blur3.gif) |![](img/coolblur.png) |  ![](img/beforeblur.png)






* Depth of field
![](img/DOF1.png) | ![](img/DOF2.png)



 Anti-aliasing |  Without-anti-aliasing
 :-------------------------:|:-------------------------: 
![](img/AA.gif) | ![](img/noAA.gif)





* Depth of field
![](img/DOF1.png) | ![](img/DOF2.png)



 Anti-aliasing |  Without-anti-aliasing
 :-------------------------:|:-------------------------: 
![](img/AA.gif) | ![](img/noAA.gif)
 


* Analysis

Stream Compaction reduces the number of active bouncing paths within one iteration, plot below illustrates this in open scene (cornell).
The thrust implementation (removeif) performs similar to the work effcicient shared memory version implementated by the author.

![](charts/1.PNG)

Stream compaction consequently significantly reduces the time consumption on computing spatial intersections of rays on objects; yet overhead of the extra
time spending on compaction itself trememdously decreases its benefit.
In comparison between the two version of compaction, shared memory provides more benefits in terms of both computing intersections and the overall CUDA running time.


![](charts/2.PNG)

In closed scene, Stream Compaction doesn't significantly the number of active bouncing paths within one iteration, plot below illustrates this in closed scene (cornell), in this case very few
rays terminated.
The thrust implementation (removeif) performs similar to the work effcicient shared memory version implementated by the author.

![](charts/3.PNG)

In closed scene, both implementation of compaction causes delay in processing; yet the overhead of the extra
time spending on compaction itself cause further delay. Lights terminated mostly (probably) due to the collision on emissive lights.
In comparison between the two version of compaction, shared memory provides more benefits in terms of both computing intersections and the overall CUDA running time.

![](charts/4.PNG)

Sorting trememdously increases the delay in processing; whereas caching first bouncing does reduces around 10 percent of the total processing time.

![](charts/5.PNG)

* CMakeLists Changes

A couple changed were made, 1. Changed CUDA compatiblity sm_20 --> sm_61; 2. Added file names from stream compaction folder; 3. Adjustments of main cmakelist file 
for fixing linking errors with Visual Studio 2013, CUDA 8 (see https://groups.google.com/forum/#!topic/cis-565-fall-2016/kMSmwhhdqFg , also includes the changes to the 
external folder).

* Togglable parameters of modes, please see common.h.
* Parameters for motion speed, camera DOF, etc, please see cornellMove.txt for reference.

* References

Same as Intructions.md