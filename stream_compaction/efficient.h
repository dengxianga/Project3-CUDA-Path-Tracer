#pragma once
#include <src/sceneStructs.h>
namespace StreamCompaction {
namespace Efficient {
    float scan(int n, int *odata, const int *idata);
	float scanOnDevice(int n, int *odata, const int *idata);
	int compact(int n, int *odata, const int *idata, float & milliscs);
	int getMyCompactIndices(int n, int *dev_indices, int * dev_bools, const int *dev_data);
	int  compactPaths(int n, PathSegment * odata_buff, int * bool_buff, int * indices_buff, PathSegment *paths);
}
}
