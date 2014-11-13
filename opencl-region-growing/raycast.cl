// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64

//Postfix a k to types and functions to avoid name collisions
typedef struct{
    float x;
    float y;
    float z;
} floatk3;

// floatk3 utilities
floatk3 crossk(floatk3 a, floatk3 b) {
    floatk3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;
    
    return c;
}

floatk3 normalizek(floatk3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;
    
    return v;
}

floatk3 add(floatk3 a, floatk3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    
    return a;
}

floatk3 scale(floatk3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;
    
    return a;
}

// Checks if position is inside the volume (floatk3 and intk3 versions)
int inside_float(floatk3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);
    
    return x && y && z;
}

// Indexing function (note the argument order)
int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}
// Trilinear interpolation
float value_at(floatk3 pos, __global unsigned char* data){
    if(!inside_float(pos)){
        return 0;
    }
    
    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);
    
    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);
    
    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;
    
    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];
    
    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;
    
    float c0 = rz*b0 + (1-rz)*b1;

    
    return c0;
}

__kernel void raycast(__global unsigned char* data,__global unsigned char* region,__global unsigned char* image){ 
    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    floatk3 camera = {.x=1000,.y=1000,.z=1000};
    floatk3 forward = {.x=-1, .y=-1, .z=-1};
    floatk3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    floatk3 right = crossk(forward, z_axis);
    floatk3 up = crossk(right, forward);

    // Creating unity lenght vectors
    forward = normalizek(forward);
    right = normalizek(right);
    up = normalizek(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    int blocks_per_row = IMAGE_DIM;

    int x = get_group_id(0) - (IMAGE_DIM/2);

    int y = get_local_id(0) - (IMAGE_DIM/2);

    // Find the ray for this pixel
    floatk3 screen_center = add(camera, forward);
    floatk3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalizek(ray);
    floatk3 pos = camera;

    // Move along the ray
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data) * (0.01 + r) ;       // Update the color based on data value, and if we're in the region
    }
    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = 255; //color > 255 ? 255 : color;
}
