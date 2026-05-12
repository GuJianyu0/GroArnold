#ifndef _UTILS_BASIC_
#define _UTILS_BASIC_

#include <vector>
#include <string>
#include <cmath>
// #include <iomanip> // for std::setprecision
// #include <unistd.h> // for exe path
// #include <stdexcept>
// #include <type_traits>
#include <iostream>
#include <cstddef>
#include <algorithm>
#include <numeric>



// #define PrecisionSetting float
#define PrecisionSetting double

// Utility function to print a single vector
template <typename T>
void print_vector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Variadic template to print multiple vectors
template <typename First, typename... Rest>
void print_vectors(const First& first, const Rest&... rest) {
    if constexpr (std::is_same_v<std::decay_t<First>, std::vector<typename First::value_type>>) {
        print_vector(first);
    } else {
        std::cout << first<< ", ";
    }
    if constexpr (sizeof...(rest) > 0) {
        print_vectors(rest...);
    }
}

template <typename T>
bool check_value_consistant_rate(
    int is_continue_running, PrecisionSetting x, PrecisionSetting x0, T info, PrecisionSetting epsilon=1e-3
){
    if( std::abs(x/x0-1.)>epsilon ){ //not consist
        if(is_continue_running>1){ //print info e.g. 10, or not print info e.g. 1
            std::cout<<"Wrong for "<<info<<":\n";
            std::cout<<x<<", "<<x0<<"\n";
            // //() Note that abs() is for integer, it is not std::abs() in cmath or math.h
            // std::cout<<abs(x/x0-1.)<<", "<<abs(x0/x-1.)<<"\n";
        }
        if(is_continue_running<=0){ //exit e.g. 0
            exit(0);
        }
        return 0;
    }else{ //consist
        return 1;
    }
}

// // Variadic template function to print multiple vectors
// template <typename... Args>
// void print_vectors(const Args&... args) {
//     (print_vector(args), ...); // Fold expression to call print_vector for each argument
// }

// Function to print a single scalar
template <typename T>
void print_scalar(const T& scalar) {
    std::cout << scalar << "\n";
}

// Variadic template function to print multiple scalars
template <typename... Args>
void print_scalars(const Args&... args) {
    (print_scalar(args), ...); // Fold expression to call print_scalar for each argument
}

// // Variadic template using fold expressions //in C++23
// template <typename... Args>
// void print_vectors(const Args&... args) {
//     ((std::cout << (std::ranges::range<Args> ? print_vector(args), "" : args) << ", "), ...);
//     std::cout << "\b\b "; // Remove trailing comma
// }

#endif