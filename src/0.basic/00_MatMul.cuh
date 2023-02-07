#ifndef MUL_MATRIX
#define MUL_MATRIX

#include <vector>

namespace basic {
    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs);

    template <class T1, class T2>
    void run(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
    
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif
