#include "fig.h"
#include <stdlib.h>
#include <assert.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

typedef struct Figure {
    unsigned char* buf;
    int num_rows;
    int num_cols;
    int padding;
    int img_width;
    int img_height;
    int tot_width;
    int tot_height;
} Figure;

Figure* figure_create(int img_width, int img_height, int num_rows, int num_cols, int padding)
{
    Figure* fig;
    int num_pixels, i;
    fig = malloc(sizeof(Figure));
    fig->num_rows = num_rows;
    fig->num_cols = num_cols;
    fig->padding = padding;
    fig->img_width = img_width;
    fig->img_height = img_height;
    fig->tot_width = img_width * num_cols + padding * (num_cols-1);
    fig->tot_height = img_height * num_rows + padding * (num_rows-1);
    num_pixels = fig->tot_width * fig->tot_height;
    fig->buf = malloc(num_pixels * sizeof(unsigned char));
    for (i = 0; i < num_pixels; i++)
        fig->buf[i] = 255; 
    return fig;
}

void figure_add(Figure* fig, int row_idx, int col_idx, float* data)
{
    int x, y, dx, dy, fig_idx, img_idx;
    x = col_idx * (fig->img_width + fig->padding);
    y = row_idx * (fig->img_height + fig->padding);
    for (dy = 0; dy < fig->img_height; dy++) {
        for (dx = 0; dx < fig->img_width; dx++) {
            fig_idx = (y+dy)*fig->tot_width+(x+dx);
            img_idx = dy*fig->img_width+dx;
            fig->buf[fig_idx] = (unsigned char)roundf(data[img_idx] * 255);
        }
    }
}

void figure_write(Figure* fig, char* filename)
{
    stbi_write_png(filename, fig->tot_width, fig->tot_height, 1, fig->buf, 0);
}

void figure_destroy(Figure* fig)
{
    free(fig->buf);
    free(fig);
}

void write_image(const char* filename, int img_width, int img_height, float* data, int scale)
{
    int i, j, k;
    int row, col;
    int num_pixels;
    unsigned char px;
    unsigned char* output;
    num_pixels = img_width * img_height;
    output = malloc(num_pixels * scale * scale * sizeof(unsigned char));
    for (i = 0; i < num_pixels; i++) {
        px = (unsigned char)roundf(data[i] * 255);
        row = i / img_height;
        col = i % img_width;
        for (j = 0; j < scale; j++) {
            for (k = 0; k < scale; k++) {
                output[row*img_height*scale*scale+j*scale*img_width+col*scale+k] = px;
            }
        }
    }
    stbi_write_png(filename, img_width*scale, img_height*scale, 1, output, 0);
    free(output);
}

