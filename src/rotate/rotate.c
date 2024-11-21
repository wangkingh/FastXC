#include "rotate.h"

void generate_rotate_matrix(double azi, double baz, double **matrix)
{
    // Calculate sin and cos values to avoid repeat calculation
    float sin_azi = sinf(azi);
    float cos_azi = cosf(azi);
    float sin_baz = sinf(baz);
    float cos_baz = cosf(baz);

    // Initialize matrix with zeros
    int i, j;
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < 9; j++)
        {
            matrix[i][j] = 0.0;
        }
    }

    // Fill in the matrix
    matrix[0][0] = -sin_azi * sin_baz; // *EE_DATA
    matrix[0][1] = -sin_azi * cos_baz; // *EN_DATA
    matrix[0][3] = -cos_azi * sin_baz; // *NE_DATA
    matrix[0][4] = -cos_azi * cos_baz; // *NN_DATA

    matrix[1][0] = -sin_azi * cos_baz; // *EE_DATA
    matrix[1][1] = sin_azi * sin_baz;  // *EN_DATA
    matrix[1][3] = -cos_azi * cos_baz; // *NE_DATA
    matrix[1][4] = cos_azi * sin_baz;  // *NN_DATA

    matrix[2][2] = sin_azi; // *EZ_DATA
    matrix[2][5] = cos_azi; // *NZ_DATA

    matrix[3][0] = -cos_azi * sin_baz; // *EE_DATA
    matrix[3][1] = -cos_azi * cos_baz; // *EN_DATA
    matrix[3][3] = sin_azi * sin_baz;  // *NE_DATA
    matrix[3][4] = sin_azi * cos_baz;  // *NN_DATA

    matrix[4][0] = -cos_azi * cos_baz; // *EE_DATA
    matrix[4][1] = cos_azi * sin_baz;  // *EN_DATA
    matrix[4][3] = sin_azi * cos_baz;  // *NE_DATA
    matrix[4][4] = -sin_azi * sin_baz; // *NN_DATA

    matrix[5][2] = cos_azi;  // *EZ_DATA
    matrix[5][5] = -sin_azi; // *NZ_DATA

    matrix[6][6] = -sin_baz; // *ZE_DATA
    matrix[6][7] = -cos_baz; // *ZN_DATA

    matrix[7][6] = -cos_baz; // *ZE_DATA
    matrix[7][7] = sin_baz;  // *ZN_DATA

    matrix[8][8] = 1; // *ZZ_DATA
}

void rotate(float **rtz_data, float **enz_data, double **rotate_matrix, int npts)
{
    int i, j, k;
    // Initialize rtz_data with zeros
    for (i = 0; i < 9; i++)
    {
        for (j = 0; j < npts; j++)
        {
            rtz_data[i][j] = 0.0;
        }
    }

    // Perform the rotation for each component
    for (k = 0; k < npts; k++) // loop for time
    {
        for (i = 0; i < 9; i++) // loop for output rtz_data
        {

            for (j = 0; j < 9; j++) // loop for input enz_data
            {
                rtz_data[i][k] += enz_data[j][k] * rotate_matrix[i][j];
            }
        }
    }
}