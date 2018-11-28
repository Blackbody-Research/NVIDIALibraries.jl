#=*
* Julia CUDA Array definitions and functions.
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

mutable struct CUDAArray
    ptr::Ptr{Nothing}
    size::Tuple{Vararg{Int}}
    freed::Bool

    # assume 'ptr' was returned from cuMemAlloc()
    function CUDAArray(ptr::Ptr{Nothing}, size::Tuple{Vararg{Int}})
        local ca::CUDAArray=new(ptr, size, false)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_array::Array{T, 1}) where T <: Number
        local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
        local result::CUresult = cuMemAlloc(device_ptr_array, sizeof(jl_array))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        local ca::CUDAArray = new(pop!(device_ptr_array), size(jl_array), false)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_matrix::Array{T, 2}) where T <: Number
        local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
        local result::CUresult = cuMemAlloc(device_ptr_array, sizeof(jl_matrix))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        local ca::CUDAArray = new(pop!(device_ptr_array), size(jl_matrix), false)
        finalizer(deallocate!, ca)
        return ca
    end
end

function deallocate!(ca::CUDAArray) #free pointers not yet deallocated
    if (!ca.freed)
        cuMemFree(ca.ptr)
        ca.freed = true
    end
    nothing
end
