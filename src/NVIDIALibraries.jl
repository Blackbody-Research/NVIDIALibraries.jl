#=*
* Load NVIDIA library definitions and functions.
*
* Copyright (C) 2018 Qijia (Michael) Jin
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*=#

module NVIDIALibraries

export CUDAArray, deallocate!

module CUDA
using Printf

include("load_cuda.jl")

let
    # Determine the latest installed CUDA toolkit version
    if (Sys.iswindows())
        local latest_cuda_version::VersionNumber = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")))
        local latest_cuda_version_string::String = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    end

    include("CUDA/" * latest_cuda_version_string * "/libcuda_" * latest_cuda_version_string * "_exports.jl")

    include("CUDA/" * latest_cuda_version_string * "/libcuda_" * latest_cuda_version_string * "_constants.jl")
    include("CUDA/" * latest_cuda_version_string * "/libcuda_" * latest_cuda_version_string * "_functions.jl")
end

include("CUDA/high_level_cuda.jl")

end # CUDA

module VectorTypes

include("cuda_vector_types.jl")

end # VectorTypes

module ComplexTypes
using ..VectorTypes

include("cuda_complex.jl")

end # ComplexTypes

module CUDARuntime
using ..CUDA
using ..VectorTypes
using Printf

# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")

include("load_cudart.jl")

let
    # Determine the latest installed CUDA toolkit version
    if (Sys.iswindows())
        local latest_cuda_version::VersionNumber = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")))
        local latest_cuda_version_string::String = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    end

    include("CUDART/" * latest_cuda_version_string * "/libcudart_" * latest_cuda_version_string * "_exports.jl")

    include("CUDART/" * latest_cuda_version_string * "/libcudart_" * latest_cuda_version_string * "_constants.jl")
    include("CUDART/" * latest_cuda_version_string * "/libcudart_" * latest_cuda_version_string * "_functions.jl")
end

include("CUDART/high_level_cudart.jl")

end # Runtime

module CUBLAS
using ..CUDA
using ..CUDARuntime
using ..VectorTypes
using ..ComplexTypes
using Printf

# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded

# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")

include("cuda_library_types.jl")

include("load_cublas.jl")

let
    # Determine the latest installed CUDA toolkit version
    if (Sys.iswindows())
        local latest_cuda_version::VersionNumber = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")))
        local latest_cuda_version_string::String = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    end

    include("CUBLAS/" * latest_cuda_version_string * "/libcublas_" * latest_cuda_version_string * "_exports.jl")

    include("CUBLAS/" * latest_cuda_version_string * "/libcublas_" * latest_cuda_version_string * "_constants.jl")
    include("CUBLAS/" * latest_cuda_version_string * "/libcublas_" * latest_cuda_version_string * "_functions.jl")
end

end # CUBLAS

module DeviceArray
using ..CUDA
using ..CUDARuntime

import Base: unsafe_copyto!, copyto!
import NVIDIALibraries.CUDA.cuLaunchKernel
import NVIDIALibraries.CUDARuntime.cudaLaunchKernel

export CUDAArray, deallocate!,
        unsafe_copyto!, copyto!,
        cuLaunchKernel, cudaLaunchKernel

include("cuda_array.jl")

end # DeviceArray

using .DeviceArray

end
