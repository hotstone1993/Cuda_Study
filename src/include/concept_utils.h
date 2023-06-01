#ifndef CONCEPT_UTILS
#define CONCEPT_UTILS

#include <concepts>

template <class T1, class T2>
concept is_type_same = std::is_same_v<T1, T2> && std::is_same_v<T2, T1>;


template <class T1>
concept is_integer = std::integral<T1> && sizeof(T1) >= 4;

#endif // CONCEPT_UTILS