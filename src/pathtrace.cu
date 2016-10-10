#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <glm/gtc/matrix_inverse.hpp>

#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "stream_compaction/common.h"
#include "stream_compaction/efficient.h"

 
#define USECOMPATION 1
#define USETHRUSTCOMPT 0
#define SORTBYKEY 0 && !USECOMPATION
#define CACHEFIRSTBOUNCE 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
 

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int * dev_remain_bounces = NULL;
static int * dev_indices4compact = NULL;
static int * dev_bools4compact = NULL;
static PathSegment * dev_paths_buff = NULL;
static int * dev_materialID_buff = NULL;
static int * dev_materialID_buff2 = NULL;
static ShadeableIntersection * dev_intersections_firstbounce = NULL;
void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_remain_bounces, pixelcount * sizeof(int));
	cudaMalloc(&dev_indices4compact, pixelcount * sizeof(int));
	cudaMalloc(&dev_bools4compact, pixelcount * sizeof(int));
	cudaMalloc(&dev_paths_buff, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_materialID_buff, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_materialID_buff2, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_intersections_firstbounce, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_firstbounce, 0, pixelcount * sizeof(ShadeableIntersection));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_remain_bounces);
	cudaFree(dev_indices4compact);
	cudaFree(dev_bools4compact);
	cudaFree(dev_paths_buff);
	cudaFree(dev_materialID_buff);
	cudaFree(dev_materialID_buff2);
	cudaFree(dev_intersections_firstbounce);
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				//when terminate, when use these values, TODO!
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].pt3 = intersect_point; //leave it here for now
		}
	}
}

__device__ void progressGatherPath(glm::vec3 * image, const PathSegment& path_segment)
{
	if (path_segment.isoff() && USECOMPATION)
	{ //refer to final gather
		image[path_segment.pixelIndex] += path_segment.color;
	}
}

__global__ void shadeMaterialAndGather(int iter
	, int num_paths
	, int depth
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, glm::vec3* image){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t <= 0.0f){
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			//thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;
			if (material.emittance > 0.0f){
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {
				scatterRay(pathSegments[idx], intersection.pt3, intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;
			}
		}
		progressGatherPath(image, pathSegments[idx]);
	}

}

// Add the current iteration's output to the overall image
__global__ void finalGatherDone(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.isoff() && USECOMPATION){
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

//add path termination bool return
struct isPathOff{
	__host__ __device__ bool operator()(const PathSegment & path_seg){
		return path_seg.isoff();
	}
};

//get the remaining bounces for streamcompaction
__global__ void getPathBounces(int n, int *obounces, const PathSegment * paths){
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < n)
	{
		PathSegment pathSegment = paths[path_index];
		obounces[path_index] = pathSegment.remainingBounces;
	}
}

__global__ void kernScatterPaths(int n, PathSegment *odata,
	const PathSegment *idata, const int *bools, const int *indices) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n) return;
	if (bools[index]){
		odata[indices[index]] = idata[index];
	}
}
__global__ void kernGetMaterialID(int n, int *obuff, const ShadeableIntersection * intersects){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index > n){
		obuff[index] = intersects[index].materialId;
	}
}
/**
* Wrapper for the __global__ call that sets up the kernel calls and does a ton
* of memory management
*/
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	//printf("traceDepth %d \n", traceDepth);
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	//add thrust ptr
	thrust::device_ptr<PathSegment> thrust_dev_path_ptr(dev_paths);
	//


	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.



	// TODO: perform one iteration of path tracing
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray failed");
	//TRY motion here     ///////////////////////////
	Geom *geoms = &(hst_scene->geoms)[0];
	glm::vec3 curTrans;
	for (int i = 0; i < hst_scene->geoms.size(); i++){
		if (geoms[i].isMoving){
			curTrans = geoms[i].translation;
			curTrans = geoms[i].translation + (geoms[i].movegoal - curTrans) *  (float)0.01;
			//printf("%f \n", (geoms[i].movegoal - curTrans).x);
			//printf("%f \n", (geoms[i].movegoal - curTrans).x *  (float)0.1 );
			//printf("%f \n", curTrans.x);
			//printf("%f \n", curTrans.x);
			//printf("%f \n", geoms[i].movegoal.x);
			geoms[i].translation = curTrans;
			geoms[i].transform = utilityCore::buildTransformationMatrix(curTrans, geoms[i].rotation, geoms[i].scale);
			geoms[i].inverseTransform = glm::inverse(geoms[i].transform);
			geoms[i].invTranspose = glm::inverseTranspose(geoms[i].transform);
		}
	}
	cudaMemcpy(dev_geoms, geoms, hst_scene->geoms.size()*sizeof(Geom), cudaMemcpyHostToDevice);
	//end motion       /////////////////////////////

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	//active paths
	int num_paths_on = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths_on + blockSize1d - 1) / blockSize1d;
		if (CACHEFIRSTBOUNCE && depth == 0 && iter == 1 || !CACHEFIRSTBOUNCE){
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();

		}
		if (CACHEFIRSTBOUNCE){
			if (depth == 0 && iter == 1){
				cudaMemcpy(dev_intersections_firstbounce, dev_intersections, num_paths_on * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else if (depth == 0&&iter > 1){//we only cache the first bounce at the very beginning and reuse it as the first bounce in other iterations (assume stationary camera and scene)
				cudaMemcpy(dev_intersections, dev_intersections_firstbounce, num_paths_on * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else{
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();
			}
		}


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		int numoffbySort = 0;
#if SORTBYKEY
		kernGetMaterialID << <numblocksPathSegmentTracing, blockSize1d >> >(num_paths_on, dev_materialID_buff, dev_intersections);
		cudaMemcpy(dev_materialID_buff2, dev_materialID_buff, num_paths_on*sizeof(int), cudaMemcpyDeviceToDevice);
		thrust::sort_by_key(thrust::device, dev_materialID_buff, dev_materialID_buff + num_paths_on, dev_paths);
		thrust::sort_by_key(thrust::device, dev_materialID_buff2, dev_materialID_buff2 + num_paths_on, dev_intersections);
/*		numoffbySort = thrust::count_if(thrust_dev_path_ptr, thrust_dev_path_ptr + num_paths_on, isPathOff());
		num_paths_on -= numoffbySort;
		printf("num_paths_on %d\n", num_paths_on)*/;
#endif
		shadeMaterialAndGather << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths_on,
			iter,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_image
			);



#if USECOMPATION
#if USETHRUSTCOMPT
		//steam compaction by thrust
		auto thrustend = thrust::remove_if(thrust::device, thrust_dev_path_ptr, thrust_dev_path_ptr + num_paths_on, isPathOff());
		num_paths_on = thrustend - thrust_dev_path_ptr;
#else
		num_paths_on = StreamCompaction::Efficient::compactPaths(num_paths_on, dev_paths_buff, dev_indices4compact, dev_bools4compact, dev_paths);
#endif
#endif
		//printf("%d \n", num_paths_on);
		
		depth++;
		iterationComplete = depth > traceDepth || num_paths_on <= 0; // DONE: should be based off stream compaction results.
		


	}

	// Assemble this iteration and apply it to the image
#if !USECOMPATION
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<< <numBlocksPixels, blockSize1d >> >(num_paths, dev_image, dev_paths);
#endif
	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
