#include <fstream>
#include <iomanip>

#ifndef ARRAY2D_H
#define ARRAY2D_H

template <typename value_t, typename index_t>
value_t **create_array2d(index_t N1, index_t N2)
{
    value_t *data = new value_t[N1 * N2];
    value_t **array = new value_t *[N1];

    index_t N = 0;
    for (index_t i = 0; i < N1; ++i)
    {
        array[i] = &data[N];
        N += N2;
    }
    return array;
}

template <typename value_t, typename index_t>
void destroy_array2d(value_t **&array)
{
    delete[] array[0];
    delete[] array;
}

template <class value_t, class index_t>
void write_txt_array2d(value_t **const z, const index_t N1, const index_t N2,
                       const std::string &fname, int buffer_size = 8096)
{
    char buffer[buffer_size]; // larger = faster (within limits)
    std::ofstream file;
    file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

    file.open(fname, std::ofstream::out);
    file << std::scientific << std::setprecision(12);
    for (index_t j = 0; j < N1; j++)
    {
        for (index_t i = 0; i < N2 - 1; i++)
        {
            file << z[j][i] << ',';
        }
        file << z[j][N2 - 1] << '\n';
    }
    file.close();
}

#endif