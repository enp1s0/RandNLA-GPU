CXX=g++
CXXFLAGS=-std=c++14 -I./src/cutf/include -O3 -I./src
CXXFLAGS+=-I./src/mateval/include -L./src/mateval/build/ -lmateval_cuda
CXXFLAGS+=-I./src/matfile/include
CXXFLAGS+=-ltmglib -llapacke -llapack -lblas -lgfortran
#CXXFLAGS+=-DTIME_BREAKDOWN
#CXXFLAGS+=-DCUTF_DISABLE_MALLOC_ASYNC
#CXXFLAGS+=-DFP16_EMULATION
OMPFLAGS=-fopenmp
NVCC=nvcc
NVCCFLAGS=$(CXXFLAGS) --compiler-bindir=$(CXX) -Xcompiler=$(OMPFLAGS)
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-lcublas -lcusolver -rdc=true -lcurand
SRCDIR=src
SRCS=$(shell find src -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')
TARGET=rsvd.test

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@

matgen_main: matgen_src/main.cpp $(SRCDIR)/input_matrix.cpp $(SRCDIR)/input_matrix_latms.cpp
	$(CXX) $+ $(CXXFLAGS) $(OMPFLAGS) -o $@ -lmpi

breakdown_aggregator: src/aggregator_main.cpp
	$(CXX) $+ -std=c++17 -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
