// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
#define DATA_SIZE (DATA_DIM * DATA_DIM * DATA_DIM) 
#define DATA_SIZE_BYTES (sizeof(unsigned char) * DATA_SIZE)

// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64
#define IMAGE_SIZE (IMAGE_DIM * IMAGE_DIM)
#define IMAGE_SIZE_BYTES (sizeof(unsigned char) * IMAGE_SIZE)


// Indexing function (note the argument order)
int region_index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);
    
    return x && y && z;
}
        
// Check if two values are similar, threshold can be changed.
int similar(__global unsigned char* data, int3 a, int3 b){
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];
    
    int i = abs(va-vb) < 1;
    return i;
}

__kernel void region(__global unsigned char* data,__global unsigned char* region,__global int* unfinished){ 
    int3 voxel;
    voxel.x = get_global_id(0);
    voxel.y = get_global_id(1);
    voxel.z = get_global_id(2);
    
    int ind = region_index(voxel.z, voxel.y, voxel.x);

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

            if(region[region_index(candidate.z, candidate.y, candidate.x)]){
                continue;
            }

            if(similar(data, voxel, candidate)){
                region[region_index(candidate.z, candidate.y, candidate.x)] = 2;
            }
        }

    }
}
