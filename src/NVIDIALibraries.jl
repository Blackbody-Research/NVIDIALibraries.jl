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

module Complex
using ..VectorTypes

include("cuda_complex.jl")

end # Complex

module CUDARuntime
using ..CUDA
using ..VectorTypes
using Printf

# CUDA runtime API is implemented over CUDA driver API

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

end # Runtime

module CUBLAS
using ..CUDA
using ..CUDARuntime
using ..VectorTypes
using ..Complex
using Printf

# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded

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

export CUDAArray, deallocate!

include("cuda_array.jl")

end # DeviceArray

using .DeviceArray

end
