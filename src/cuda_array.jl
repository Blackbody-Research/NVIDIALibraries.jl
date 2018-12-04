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
    is_device::Bool # or host memory

    # assume 'ptr' was returned from cuMemAlloc()
    function CUDAArray(ptr::Ptr{Nothing}, size::Tuple{Vararg{Int}}, is_device::Bool)
        local ca::CUDAArray=new(ptr, size, false, is_device)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_vector::Array{T, 1}) where T
        local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]

        # allocate new array in device memory
        local result::CUresult = cuMemAlloc(device_ptr_array, sizeof(jl_vector))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        # copy data to the array in device memory
        result = cuMemcpyHtoD(device_ptr_array[1], Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, jl_vector)), sizeof(jl_vector))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        local ca::CUDAArray = new(pop!(device_ptr_array), size(jl_vector), false, true)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_matrix::Array{T, 2}) where T
        local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]

        # allocate new array in device memory
        local result::CUresult = cuMemAlloc(device_ptr_array, sizeof(jl_matrix))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        # copy data to the array in device memory
        result = cuMemcpyHtoD(device_ptr_array[1], Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, jl_matrix)), sizeof(jl_matrix))

        if (result != CUDA_SUCCESS)
            error("CUDAArray error: ", unsafe_string(cuGetErrorString(result)))
        end

        local ca::CUDAArray = new(pop!(device_ptr_array), size(jl_matrix), false, true)
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
