#ifndef BMP
#define BMP

#include <stdint.h>

struct bmp_id{
  uint8_t magic1;
  uint8_t magic2;
};

struct bmp_header{
  uint32_t file_size;
  uint16_t creator1;
  uint16_t creator2;
  uint32_t pixel_offset;
};

struct bmp_dib_header{
  uint32_t header_size;
  int32_t width;
  int32_t height;
  uint16_t num_planes;
  uint16_t bit_pr_pixel;
  uint32_t compress_type;
  uint32_t data_size;
  int32_t hres;
  int32_t vres;
  uint32_t num_colors;
  uint32_t num_imp_colors;
};

void write_bmp(unsigned char* data, int width, int height);
unsigned char* read_bmp(char* filename);

#endif
