
int3 voxel;
voxel.x = blockIdx.x * blockDim.x + threadIdx.x;
voxel.y = blockIdx.y * blockDim.y + threadIdx.y;
voxel.z = blockIdx.z * blockDim.z + threadIdx.z;

int ind = index(voxel.z, voxel.y, voxel.x);

if(region[ind] == 2){
    //Race conditions should not matter, as we only write 1s, and if one of them gets through it's enough
    *unfinished = 1; 
    region[ind] = 1;

    int dx[6] = {-1,1,0,0,0,0};
    int dy[6] = {0,0,-1,1,0,0};
    int dz[6] = {0,0,0,0,-1,1};



    for(int n = 0; n < 6; n++){
        int3 candidate;
        candidate.x = voxel.x + dx[n];
        candidate.y = voxel.y + dy[n];
        candidate.z = voxel.z + dz[n];

        if(!inside(candidate)){
            continue;
        }

        if(region[index(candidate.z, candidate.y, candidate.x)]){
            continue;
        }

        if(similar(data, voxel, candidate)){
            region[index(candidate.z, candidate.y, candidate.x)] = 2;
        }
    }

}
