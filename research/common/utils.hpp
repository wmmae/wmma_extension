#ifndef __WMMAE_RESEARCH_UTILS_HPP__
#define __WMMAE_RESEARCH_UTILS_HPP__
#include <sstream>

#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR(status) cuda_check_error(status, __FILE__, __LINE__, __func__)
#endif


inline void cuda_check_error(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss<< cudaGetErrorString(error);
		ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}

#endif
