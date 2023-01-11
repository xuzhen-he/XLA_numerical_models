#
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. vec_copy.cpp -o bin/vec_copy
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. vec_scale.cpp -o bin/vec_scale
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. vec_axpy.cpp -o bin/vec_axpy
g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. vec_xpxpy.cpp -o bin/vec_xpxpy