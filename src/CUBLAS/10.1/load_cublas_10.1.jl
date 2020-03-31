#=*
* Load CUBLAS v10.1 library
*
* Copyright (C) 2018-2020 Blackbody Research LLC
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
    const libcublas = Libdl.find_library([@sprintf("cublas%i_101", Sys.WORD_SIZE)], ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin"])
elseif (Sys.isapple())
    const libcublas = Libdl.find_library(["libcublas.10.1"], ["/Developer/NVIDIA/CUDA-10.1/lib/"])
elseif (Sys.islinux())
    if (Sys.WORD_SIZE == 32)
        const libcublas = Libdl.find_library(["libcublas"], ["/usr/local/cuda-10.1/lib/"])
    elseif (Sys.WORD_SIZE == 64)
        const libcublas = Libdl.find_library(["libcublas"], ["/usr/local/cuda-10.1/lib64/"])
    end
end
