#ifndef FIG_H
#define FIG_H

typedef struct Figure Figure;

Figure*     figure_create(int img_width, int img_height, int num_rows, int num_cols, int padding);
void        figure_add(Figure* fig, int row_idx, int col_idx, float* data);
void        figure_write(Figure* fig, char* filename);
void        figure_destroy(Figure* fig);
void        write_image(const char* filename, int img_width, int img_height, float* data, int scale);

#endif
