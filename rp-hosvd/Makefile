CXX=g++
CXXFLAGS=-lcutensor
CXXFLAGS+=-ltmglib -llapacke -llapack -lblas -lgfortran
#CXXFLAGS+=-DTIME_BREAKDOWN
#CXXFLAGS+=-DCUTF_DISABLE_MALLOC_ASYNC
#CXXFLAGS+=-DFP16_EMULATION
CXXFLAGS+=-DRUN_REFERENCE_FUNCTIONS
OMPFLAGS=-fopenmp
NVCC=nvcc

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -maxdepth 1 -name '*.cu' -o -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
HEADERS=$(shell find $(SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

COMMON_SRCDIR=../common/src
COMMON_SRCS=$(shell find $(COMMON_SRCDIR) -maxdepth 1 -name '*.cu' -o -name '*.cpp')
COMMON_OBJDIR=common_objs
COMMON_OBJS=$(subst $(COMMON_SRCDIR),$(COMMON_OBJDIR), $(COMMON_SRCS))
COMMON_OBJS:=$(subst .cpp,.o,$(COMMON_OBJS))
COMMON_OBJS:=$(subst .cu,.o,$(COMMON_OBJS))
COMMON_HEADERS=$(shell find $(COMMON_SRCDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

CXXFLAGS+=-std=c++14 -O3 -I./$(SRCDIR) -I$(COMMON_SRCDIR)
CXXFLAGS+=-I$(COMMON_SRCDIR)/cutf/include
CXXFLAGS+=-I$(COMMON_SRCDIR)/cuta/include
CXXFLAGS+=-I$(COMMON_SRCDIR)/mateval/include -L$(COMMON_SRCDIR)/mateval/build/ -lmateval_cuda
CXXFLAGS+=-I$(COMMON_SRCDIR)/matfile/include
CXXFLAGS+=-I$(COMMON_SRCDIR)/shgemm/include -L$(COMMON_SRCDIR)/shgemm/build -lshgemm
CXXFLAGS+=-I$(COMMON_SRCDIR)/curand_fp16/include -L$(COMMON_SRCDIR)/curand_fp16/build -lcurand_fp16

NVCCFLAGS=$(CXXFLAGS) --compiler-bindir=$(CXX) -Xcompiler=$(OMPFLAGS)
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-lcublas -lcusolver -rdc=true -lcurand

OBJS+=$(COMMON_OBJS)
HEADERS+=$(COMMON_HEADERS)

TARGET=rp-hosvd.test

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@

$(COMMON_SRCDIR)/%.cpp: $(COMMON_SRCDIR)/%.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(COMMON_OBJDIR)/%.o: $(COMMON_SRCDIR)/%.cpp
	[ -d $(COMMON_OBJDIR) ] || mkdir $(COMMON_OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@

matgen_main: matgen_src/main.cpp $(SRCDIR)/input_matrix.cpp $(SRCDIR)/input_matrix_latms.cpp
	$(CXX) $+ $(CXXFLAGS) $(OMPFLAGS) -o $@ -lmpi

breakdown_aggregator: aggr_src/aggregator_main.cpp
	$(CXX) $+ -std=c++17 -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
