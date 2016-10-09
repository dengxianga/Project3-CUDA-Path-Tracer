#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define NUM_BANKS 16  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

 

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
__global__ void upSweep(int offset, int n,   int *idata){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >=n) return;
	int tmp=(offset << 1);
	if (index % tmp==0){
		if (index + tmp <=n){ 
			idata[index+tmp-1] += idata[index+offset-1]  ;		 
		}
	}
}

__global__ void downSweep(int offset, int n,  int *idata){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >=n) return;
	int tmp=(offset << 1);
	if (index % tmp==0){

		if (index + tmp <= n){
			int t = idata[index + offset -1];
			idata[index+offset-1] = idata[index+ tmp -1];
			idata[index+ tmp -1] += t ;
		}
 
	}
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
float scan(int n, int *odata, const int *idata) {
	cudaEvent_t t0, t2;
	cudaEventCreate(&t0);
	cudaEventCreate(&t2); 
 
	float milliscs = 0.0f;
	float tmpt;
    // TODO
    //printf("TODO\n");
	int levels_max = ilog2ceil(n);
	int n_max= 1 << levels_max;

	dim3 numblocks(std::ceil((double) n_max / blockSize));
	int* idata_buff;
	//allocate more space than needed
	cudaMalloc((void**)&idata_buff, n_max*sizeof(int)); 	
		checkCUDAError("cudaMalloc-idata_buff-  failed!");	
	//reset all to zeros
    cudaMemset(idata_buff, 0, n_max*sizeof(int));
		checkCUDAError("cudaMemset-idata_buff-  failed!");	

	/// CPU -->GPU
	cudaMemcpy(idata_buff,idata,n*sizeof(int),cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy-idata_buff-failed");

	cudaEventRecord(t0);

	//upsweep
	for (int level=0; level <= levels_max-1; level++){
		upSweep<<<numblocks,blockSize>>>(1<<level, n_max, idata_buff);
	}

	cudaEventRecord(t2);	
	cudaEventSynchronize(t2);
	cudaEventElapsedTime(&tmpt, t0, t2);	 
	milliscs += tmpt;


	//downsweep
	//set root x[n-1]=0
	//idata_buff[n_max-1]=0;
	cudaMemset(idata_buff+n_max-1, 0,  sizeof(int));
		
	cudaEventRecord(t0);

	for (int level=levels_max-1; level >=0 ; level--){
		downSweep<<<numblocks,blockSize>>>(1<<level, n_max, idata_buff);
	}

	cudaEventRecord(t2);
	cudaEventSynchronize(t2);	
	cudaEventElapsedTime(&tmpt, t0, t2);
	milliscs += tmpt;

	/// GPU --> CPU
	cudaMemcpy(odata, idata_buff, n*sizeof(int),cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy-odata-failed");
	cudaFree(idata_buff);
	return milliscs;
}


float scanOnDevice(int n, int *odata, const int *idata) {
 
	 
	int levels_max = ilog2ceil(n);
	int n_max = 1 << levels_max;

	dim3 numblocks(std::ceil((double)n_max / blockSize));
	int* idata_buff;
	//allocate more space than needed
	cudaMalloc((void**)&idata_buff, n_max*sizeof(int));
	checkCUDAError("cudaMalloc-idata_buff-  failed!");
	//reset all to zeros
	cudaMemset(idata_buff, 0, n_max*sizeof(int));
	checkCUDAError("cudaMemset-idata_buff-  failed!");

	/// GPU -->GPU
	cudaMemcpy(idata_buff, idata, n*sizeof(int), cudaMemcpyDeviceToDevice);
	checkCUDAError("cudaMemcpy-idata_buff-failed");

 

	//upsweep
	for (int level = 0; level <= levels_max - 1; level++){
		upSweep << <numblocks, blockSize >> >(1 << level, n_max, idata_buff);
	}
	  


	//downsweep
	//set root x[n-1]=0
	//idata_buff[n_max-1]=0;
	cudaMemset(idata_buff + n_max - 1, 0, sizeof(int));
	 

	for (int level = levels_max - 1; level >= 0; level--){
		downSweep << <numblocks, blockSize >> >(1 << level, n_max, idata_buff);
	}

 
	/// GPU --> GPU
	cudaMemcpy(odata, idata_buff, n*sizeof(int), cudaMemcpyDeviceToDevice);
	checkCUDAError("cudaMemcpy-odata-failed");
	cudaFree(idata_buff);
	return 0;
}
/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata, float &milliscs) {
	cudaEvent_t t0, t2;
	cudaEventCreate(&t0);
	cudaEventCreate(&t2);

	milliscs = 0.0f;
	float tmpt;


    int n_remaing=0;
	int * idata_buff;
	int * odata_buff;
	int * bool_buff;
	int * indices_buff;

	dim3 numblocks(std::ceil((double) n/blockSize));
	//
	cudaMalloc((void**)&idata_buff,n * sizeof(int));
		checkCUDAError("cudaMalloc-idata_buff-failed");
	cudaMalloc((void**)&odata_buff,n * sizeof(int));
		checkCUDAError("cudaMalloc-odata_buff-failed");
	cudaMalloc((void**)&bool_buff,n * sizeof(int));
		checkCUDAError("cudaMalloc-odata_buff-failed");
	cudaMalloc((void**)&indices_buff,n * sizeof(int));
		checkCUDAError("cudaMalloc-odata_buff-failed");

	cudaMemcpy(idata_buff, idata, n* sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy-idata_buff-failed");
	cudaMemcpy(odata_buff, odata, n* sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy-odata_buff-failed");
	
	cudaEventRecord(t0);
	//produce the indices
	StreamCompaction::Common::kernMapToBoolean<<<numblocks, blockSize>>> ( n, bool_buff, idata_buff);

	scan  (n, indices_buff, bool_buff);

	StreamCompaction::Common::kernScatter<<<numblocks, blockSize>>>( n, odata_buff, idata_buff,  bool_buff,  indices_buff);
	
	cudaEventRecord(t2);
	cudaEventSynchronize(t2);	
	cudaEventElapsedTime(&tmpt, t0, t2);
	milliscs += tmpt;

	//GPU-->CPU
	cudaMemcpy(odata,odata_buff,n*sizeof(int),cudaMemcpyDeviceToHost);

	//for (int i =0; i< n; i++){
	//	n_remaing+=bool_buff[i];
	//}
	cudaMemcpy(&n_remaing,indices_buff+n-1,sizeof(int),cudaMemcpyDeviceToHost);
	int extra;
	cudaMemcpy(&extra, bool_buff + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(idata_buff);
	cudaFree(odata_buff);
	cudaFree(bool_buff);
	cudaFree(indices_buff);
	return n_remaing + extra;
}
 

__global__ void kernMapPathsToBoolean(int n, int *bools, const PathSegment *paths) {
	// TODO
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		bools[index] = paths[index].remainingBounces != 0;
	}
}
__global__ void kernPathsScatter(int n, PathSegment *odata,
	const PathSegment *idata, const int *bools, const int *indices) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= n) return;
	if (bools[index]){
		odata[indices[index]] = idata[index];
	}
}
 



 

// see http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__global__ void preScanShared(int n, int * g_odata, const int* g_idata){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	 
	extern __shared__ float s_idata[];  // allocated on invocation  
	int thid = threadIdx.x;
	int offset = 1;
	//int ai0 = thid;
	//int bi0 = thid + (n / 2);
	//int bankOffsetA = CONFLICT_FREE_OFFSET(ai0);
	//int bankOffsetB = CONFLICT_FREE_OFFSET(bi0);
 
	s_idata[2 * thid] = g_idata[2 * index]; // load input into shared memory  
	s_idata[2 * thid + 1] = g_idata[2 * index + 1];
	//s_idata[ai0 + bankOffsetA] = g_idata[2 * index]; // load input into shared memory  
	//s_idata[bi0 + bankOffsetB] = g_idata[2 * index + 1];
	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			////banking conflict
			//ai += CONFLICT_FREE_OFFSET(ai);
			//bi += CONFLICT_FREE_OFFSET(bi);

			s_idata[bi] += s_idata[ai];
		}
		offset *= 2;
		
	}
	//if (thid == 0) { s_idata[n - 1] = 0; } // clear the last element  
	if (thid == 0) { s_idata[n - 1+CONFLICT_FREE_OFFSET(n - 1)] = 0; } // clear the last element  
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			//banking conflict
			//ai += CONFLICT_FREE_OFFSET(ai);
			//bi += CONFLICT_FREE_OFFSET(bi);
			float t = s_idata[ai];
			s_idata[ai] = s_idata[bi];
			s_idata[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * index] = s_idata[2 * thid]; // write results to device memory  
	g_odata[2 * index + 1] = s_idata[2 * thid + 1];
	//banking conflict	 
	//g_odata[2 * index] = s_idata[ai0 + bankOffsetA]; // write results to device memory  
	//g_odata[2 * index + 1] = s_idata[bi0 + bankOffsetB];
}
 
//http://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
__global__ void storeBlockSums(int n, int *sum_buff, const int * odata, const int * idata){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		int offset = (index + 1)*blockSize - 1;
		sum_buff[index] = odata[offset] + idata[offset];
	}
}
__global__ void sumBuff2Blocks(int n, int *odata, int * sumbuff){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		odata[index] += sumbuff[blockIdx.x];
	}
}
void scanMultiBlocks(int n, int * odata, const int * idata){
	int numblocks(std::ceil((double)n / blockSize)); 
	//int numblockshalf = numblocks / 2;
	int nmax = numblocks*blockSize;
 
	//kernel scan here
	//if (numblocks == 1){
	preScanShared << <numblocks, blockSize / 2, blockSize*sizeof(int) >> >(blockSize, odata, idata);
		//return;
	//}
	
	if (numblocks > 1){
		int numblocksum(std::ceil((double)numblocks / blockSize)); 
		int * dev_sum_buff;
		//int * dev_odata_buff;
		int *dev_scan_sum_buff;
		cudaMalloc((void**)&dev_sum_buff, numblocks*sizeof(int));
		cudaMalloc((void**)&dev_scan_sum_buff, numblocks *sizeof(int));

		
		storeBlockSums << <numblocksum, blockSize >> >(numblocks, dev_sum_buff, odata, idata);
		//resursive scan here
		//scanMultiBlocks(numblocksum, dev_scan_sum_buff, dev_sum_buff);
		scanMultiBlocks(numblocks, dev_scan_sum_buff, dev_sum_buff);
		sumBuff2Blocks <<< numblocks, blockSize >> > (nmax, odata, dev_scan_sum_buff);

		//free buff here
		cudaFree(dev_sum_buff);
		cudaFree(dev_scan_sum_buff);
	}

	
}
  

int compactPaths(int n, PathSegment * odata_buff, int * bool_buff, int * indices_buff, PathSegment *paths){
 


	int n_remaing = 0;
 
	dim3 numblocks(std::ceil((double)n / blockSize));
	//
 
 

	 
	cudaMemcpy(odata_buff, paths, n* sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	checkCUDAError("cudaMemcpy-odata_buff-failed");

	 ;
	//produce the indices
	kernMapPathsToBoolean << <numblocks, blockSize >> > (n, bool_buff, paths);

	 //scanOnDevice(n, indices_buff, bool_buff); 
	scanMultiBlocks(n, indices_buff, bool_buff);
	//int levels_max = ilog2ceil(n);
	//int n_max = 1 << levels_max;
	//dim3 numblocksmax(std::ceil((double)n_max / blockSize));
	//kernScanShared << <numblocksmax, blockSize >> >(n, indices_buff, bool_buff);

	kernPathsScatter << <numblocks, blockSize >> >(n, odata_buff, paths, bool_buff, indices_buff);

 

	//GPU-->GPU
	cudaMemcpy(paths, odata_buff, n*sizeof(PathSegment), cudaMemcpyDeviceToDevice);

 
	cudaMemcpy(&n_remaing, indices_buff + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	int extra;
	cudaMemcpy(&extra, bool_buff + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
	 
 
	return n_remaing + extra;
}
}
}
