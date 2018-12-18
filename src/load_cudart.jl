#=*
* Load CUDA runtime library
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

# find the most up to date version of CUDA Toolkit installed
local latest::VersionNumber

if (Sys.iswindows())
    @assert (length(readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\")) != 0)
    latest = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\")))
    const libcudart = Libdl.find_library([@sprintf("cudart%i_%i%i", Sys.WORD_SIZE, latest.major, latest.minor)],
                                        [@sprintf("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%i.%i\\bin", latest.major, latest.minor)])
elseif (Sys.isapple())
    @assert (length(readdir("/Developer/NVIDIA/")) != 0)
    latest = reduce(max,
                    map(VersionNumber,
                        map((function(name::String)
                                return name[6:end]
                            end),
                            readdir("/Developer/NVIDIA/"))))
    const libcudart = Libdl.find_library([@sprintf("libcudart.%i.%i", latest.major, latest.minor)],
                                        [@sprintf("/Developer/NVIDIA/CUDA-%i.%i/lib/", latest.major, latest.minor)])
end
