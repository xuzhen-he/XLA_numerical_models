#
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. mat_copy.cpp -o bin/mat_copy
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. mat_scale.cpp -o bin/mat_scale
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. mat_axpy.cpp -o bin/mat_axpy
g++ -fPIC -Wall -Wextra -O3 -fopenmp -fopt-info-vec -march=native -I../.. -I.. mat_xpxpy.cpp -o bin/mat_xpxpy