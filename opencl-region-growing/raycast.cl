// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64

__kernel raycast(__global unsigned char* data,__global unsigned char* region,__global unsigned char* image){ 
    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = float(3.14)/4;
    float pixel_width = tan(fov/float(2.0))/(IMAGE_DIM/2);
    float step_size = 0.5;

    int blocks_per_row = IMAGE_DIM/blockDim.x;

    int x 
        = (blockIdx.x % blocks_per_row) * blockDim.x 
        + threadIdx.x 
        - (IMAGE_DIM/2);

    int y 
        = blockIdx.x/blocks_per_row  
        - (IMAGE_DIM/2);

    // Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    // Move along the ray
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data)*(float(0.01) + r) ;       // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}
