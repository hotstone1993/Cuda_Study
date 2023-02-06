#ifndef MUL_MATRIX_MT
#define MUL_MATRIX_MT

#include <vector>

namespace basic_mt {
    template <class T1, class T2>
    void run(std::vector<T1*>& inputs, std::vector<T2*>& outputs);

    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // MUL_MATRIX_MT