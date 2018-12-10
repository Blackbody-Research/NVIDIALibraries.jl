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
        local device_pointer_array::Array{Ptr{Nothing}, 1} = [C_NULL]
        cudaMalloc(device_pointer_array, sizeof(jl_array))

        # allocate new array in device memory
        local devptr::Ptr{Nothing} = pop!(device_pointer_array)

        # copy data to the array in device memory
        cudaMemcpy(devptr, jl_array, sizeof(jl_array), cudaMemcpyHostToDevice)

        local ca::CUDAArray = new(devptr, size(jl_array), false, true, T)
        finalizer(deallocate!, ca)
        return ca
    end
end

# free pointers not yet deallocated
function deallocate!(ca::CUDAArray)
    if (!ca.freed && ca.is_device)
        cudaFree(ca.ptr)
        ca.freed = true
    end
    nothing
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::Array{T}, doffset::Csize_t, src::CUDAArray, soffset::Csize_t, n::Integer)::Array where T
    if (src.is_device)
        cudaMemcpy(dst + doffset, src.ptr + soffset, sizeof(T) * n, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, dst)) + doffset,
            src.ptr + soffset,
            Csize_t(sizeof(T) * n))
    end
    return dst
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_copyto!(dst::CUDAArray, doffset::Csize_t, src::Array{T}, soffset::Csize_t, n::Integer)::Array where T
    if (dst.is_device)
        cudaMemcpy(dst.ptr + doffset, src + soffset, sizeof(T) * n, cudaMemcpyHostToDevice)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            dst.ptr + doffset,
            Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, src)) + soffset,
            Csize_t(sizeof(T) * n))
    end
    return dst
end

unsafe_copyto!(dst::Array, doffset::Integer, src::CUDAArray, soffset::Integer, n::Integer) = unsafe_copyto!(dst, Csize_t(doffset), src, Csize_t(soffset), n)
unsafe_copyto!(dst::CUDAArray, doffset::Integer, src::Array, soffset::Integer, n::Integer) = unsafe_copyto!(dst, Csize_t(doffset), src, Csize_t(soffset), n)

function unsafe_copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    local src_byte_size::Csize_t = sizeof(src.element_type) * reduce(*, src.size)
    if (src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToDevice)
    elseif (!src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyHostToDevice)
    elseif (src.is_device && !dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t), dst.ptr, src.ptr, src_byte_size)
    end
    return dst
end

function copyto!(dst::Array{T}, src::CUDAArray)::Array where T
    @assert (size(dst) == src.size)
    if (src.is_device)
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyDeviceToHost)
    else
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyDeviceToHost)
    end
    return dst
end

function copyto!(dst::CUDAArray, src::Array{T})::CUDAArray where T
    @assert (dst.size == size(src))
    if (dst.is_device)
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToDevice)
    else
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToDevice)
    end
    return dst
end

function copyto!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    @assert ((sizeof(dst.element_type) * reduce(*, dst.size))
            >= (sizeof(src.element_type) * reduce(*, src.size)))
    return unsafe_copyto!(dst, src)
end

function _cast_cudaarray_args(x::T) where T
    if (T <: CUDAArray)
        return x.ptr
    else
        return x
    end
end

@inline function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Tuple{Vararg{DataType}}, args...; kwargs...)
    return cuLaunchKernel(func, grid, block, Tuple{types...}, map(_cast_cudaarray_args, args)...; kwargs...)
end

@generated function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Type, args...; stream::CUstream = CUstream(C_NULL), shmem::Integer = 0, kwargs...)
    local result_expr::Expr = Expr(:block)

    push!(result_expr.args, Base.@_inline_meta)

    args_types = types.parameters[1].parameters
    args_syms = Array{Symbol, 1}(undef, length(args))
    args_ptrs = Array{Symbol, 1}(undef, length(args))

    for i in 1:length(args)
        # assign safely referenced data to corresponding symbol
        args_syms[i] = gensym()
        push!(result_expr.args, :($(args_syms[i]) = Base.cconvert($(args_types[i]), args[$i])))
        
        # generate julia expressions to obtain 
        args_ptrs[i] = gensym()
        push!(result_expr.args, :($(args_ptrs[i]) = Base.unsafe_convert($(args_types[i]), $(args_syms[i]))))
    end
    
    append!(result_expr.args, (quote
        GC.@preserve $(args_syms...) begin
            arguments::Array{Any, 1} = [$(args_ptrs...)]
            result::CUresult = cuLaunchKernel(func,
                                grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                Cuint(shmem),
                                stream,
                                Base.unsafe_convert(Ptr{Ptr{Nothing}}, arguments),
                                Ptr{Ptr{Nothing}}(C_NULL))
            @assert (result == CUDA_SUCCESS) ("cuLaunchKernel() error: " * cuGetErrorName(result))
        end
    end).args)

    return result_expr
end


