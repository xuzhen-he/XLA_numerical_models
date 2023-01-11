#ifndef KAHAN_SUMMATION_H
#define KAHAN_SUMMATION_H

template <typename value_t, typename index_t>
value_t Kahan_summation(value_t *y, index_t vec_size)
{
    value_t sum = 0.0;
    value_t c = 0.0;
    for (index_t j = 0; j < vec_size; j++)
    {
        value_t b = y[j] - c;
        value_t t = sum + b;
        c = (t - sum) - b;
        sum = t;
    }
    return sum;
}

#endif