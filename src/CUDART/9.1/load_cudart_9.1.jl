#=*
* Load CUDA runtime v9.1 library
*
* Copyright (C) 2018 Blackbody Research LLC
*       Author: Qijia (Michael) Jin
*
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

using Libdl
using Printf

if (Sys.iswindows())
    const libcudart = Libdl.find_library([@sprintf("cudart%i_91", Sys.WORD_SIZE)], ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\bin"])
elseif (Sys.isapple())
    const libcudart = Libdl.find_library(["libcudart.9.1"], ["/Developer/NVIDIA/CUDA-9.1/lib/"])
elseif (Sys.islinux())
    if (Sys.WORD_SIZE == 32)
        const libcudart = Libdl.find_library(["libcudart"], ["/usr/local/cuda-9.1/lib/"])
    elseif (Sys.WORD_SIZE == 64)
        const libcudart = Libdl.find_library(["libcudart"], ["/usr/local/cuda-9.1/lib64/"])
    end
end
