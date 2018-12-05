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
    element_type::DataType

    # assume 'ptr' was returned from cuMemAlloc()
    function CUDAArray(ptr::Ptr{Nothing}, size::Tuple{Vararg{Int}}, is_device::Bool, element_type::DataType)
        local ca::CUDAArray=new(ptr, size, false, is_device, element_type)
        finalizer(deallocate!, ca)
        return ca
    end

    function CUDAArray(jl_array::Array{T}) where T
        local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]

        # allocate new array in device memory
        local devptr::CUdeviceptr = cuMemAlloc(sizeof(jl_array))

        # copy data to the array in device memory
        cuMemcpyHtoD(devptr, 0, jl_array, sizeof(jl_array))

        local ca::CUDAArray = new(devptr, size(jl_array), false, true, T)
        finalizer(deallocate!, ca)
        return ca
    end
end

# free pointers not yet deallocated
function deallocate!(ca::CUDAArray)
    if (!ca.freed)
        cuMemFree(ca.ptr)
        ca.freed = true
    end
    nothing
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::Array{T}, doffset::Integer, src::CUDAArray, soffset::Integer, n::Integer)::Array where T
    cuMemcpyDtoH(dst, doffset, src, soffset, sizeof(T) * n)
    return dst
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::CUDAArray, doffset::Csize_t, src::Array{T}, soffset::Csize_t, n::Integer)::Array where T
    if (dst.is_device)
        cuMemcpyHtoD(dst, doffset, src, soffset, sizeof(T) * n)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            dst.ptr + doffset,
            Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, src)) + srcOffset,
            Csize_t(sizeof(T) * n))
    end
    return dst
end

unsafe_copyto!(dst::CUDAArray, doffset::Integer, src::Array, soffset::Integer, n::Integer)::Array = unsafe_copyto!(dst, Csize_t(doffset), src, Csize_t(soffset), n)

function unsafe_copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    local src_byte_size::Csize_t = sizeof(src.element_type) * reduce(*, src.size)
    if (src.is_device && dst.is_device)
        cuMemcpyDtoD(dst, 0, src, 0, src_byte_size)
    elseif (!src.is_device && dst.is_device)
        cuMemcpyHtoD(dst, 0, src, 0, src_byte_size)
    elseif (src.is_device && !dst.is_device)
        cuMemcpyDtoH(dst, 0, src, 0, src_byte_size)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t), dst.ptr, src.ptr, src_byte_size)
    end
    return dst
end

function copyto!(dst::Array{T}, src::CUDAArray)::Array where T
    @assert (size(dst) == src.size)
    cuMemcpyDtoH(dst, 0, src, 0, sizeof(dst))
    return dst
end

function copyto!(dst::CUDAArray, src::Array{T})::CUDAArray where T
    @assert (dst.size == size(src))
    cuMemcpyHtoD(dst, 0, src, 0, sizeof(src))
    return dst
end

function copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    @assert ((sizeof(dst.element_type) * reduce(*, dst.size))
            >= (sizeof(src.element_type) * reduce(*, src.size)))
    return unsafe_copyto!(dst, src)
end



