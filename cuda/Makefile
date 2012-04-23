
# Fill in the name of the output binary here
target    := nmf

# List of sources, with .c, .cu, and .cc extensions
sources   := matrix.cu nmf.cu

# Other things that need to be built, e.g. .cubin files
extradeps := 


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

#-m32 keeps code 32bit (required in snow leopard for now)
opt  ?= -O3 #-m32
dbg  ?= 
warn ?= -Wall -Werror


# This is where the cuda runtime libraries and includes reside on the
# berkeley millennium/rad clusters. It may be different on your machine.

#cudaroot  := /usr/mill/pkg/cuda
cudaroot  := /usr/local/cuda


#----- C compilation options ------
gcc        := /usr/bin/gcc
cflags     += $(opt) $(dbg) $(warn)
clib_paths :=

cinc_paths := -I $(cudaroot)/include
clibraries := -lcuda -lcudart -lblas


#----- C++ compilation options ------
gpp         := /usr/bin/g++
ccflags     += $(opt) $(dbg) $(warn)
cclib_paths :=
ccinc_paths := 
cclibraries :=

#----- CUDA compilation options -----

nvcc        := $(cudaroot)/bin/nvcc
cuflags     += $(opt) $(dbg) --host-compilation=C
culib_paths := -L$(cudaroot)/lib
cuinc_paths := -I$(cudaroot)/include
culibraries := -lcublas


lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths)
libraries   := $(culibraries) $(cclibraries) $(clibraries)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c ,$(sources))
ccsources := $(filter %.cc,$(sources))
cusources := $(filter %.cu,$(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.c,%.o ,$(csources))  \
           $(patsubst %.cu,%.o,$(cusources)) \
           $(patsubst %.cc,%.o,$(ccsources))


#----- Build rules ------

$(target): $(extradeps) 


$(target): $(objects) 
	$(gcc) -o $@ $(lib_paths) $(libraries) $(objects)

bench: bench.o matrix.o 
	$(gcc) -o $@ $(lib_paths) $(libraries) bench.o matrix.o

%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(cuinc_paths) -o $@ 

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $^

%.o: %.cc
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@

%.o: %.c
	$(gcc) -c $^ $(cflags) $(cinc_paths) -o $@

clean:
	rm -f *.o $(target) bench makefile.*dep *.linkinfo test

