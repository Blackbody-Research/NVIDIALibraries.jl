#=*
* High level CUDA functions
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

function cuGetErrorString(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorString(error, verbose_error)
    @assert (result == CUDA_SUCCESS)
    return unsafe_string(pop!(verbose_error))
end

function cuGetErrorName(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorName(error, verbose_error)
    @assert (result == CUDA_SUCCESS) ("cuGetErrorName() error: " * cuGetErrorString(result))
    return unsafe_string(pop!(verbose_error))
end

function cuMemAlloc(bytesize::Csize_t)::CUdeviceptr
    local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuMemAlloc(device_ptr_array, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemAlloc() error: " * cuGetErrorString(result))
    return pop!(device_ptr_array)
end

