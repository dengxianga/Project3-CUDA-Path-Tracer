#pragma once
#include <src/sceneStructs.h>
#define DEVICE_SHARED_MEMORY  8
namespace StreamCompaction {
namespace Efficient {
    float scan(int n, int *odata, const int *idata);
	float scanOnDevice(int n, int *odata, const int *idata);
	int compact(int n, int *odata, const int *idata, float & milliscs); 
	int  compactPaths(int n, PathSegment * odata_buff, int * bool_buff, int * indices_buff, PathSegment *paths);
}
}
