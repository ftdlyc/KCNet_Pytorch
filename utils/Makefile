VPATH=.
OBJDIR=./obj/
SLIB=libkcnet_utils.so

CC=gcc
CXX=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
LDFLAGS= -L/usr/local/cuda/lib64 \
         -lm -pthread -lcuda -lcudart -lcublas -lcurand -lcudnn -lstdc++
INCLUDES= -Iinclude \
          -I/usr/local/cuda/include \
          -I/home/lyc/torch4/lib/python2.7/site-packages/torch/lib/include \
          -I/home/lyc/torch4/lib/python2.7/site-packages/torch/lib/include/TH \
          -I/home/lyc/torch4/lib/python2.7/site-packages/torch/lib/include/THC \
          -I/home/lyc/torch4/include/python2.7
OPTS=-O3 -DNDEBUG
CXXFLAGS= -w -std=c++11 -g -fPIC
NVCCFLAGS= -fPIC
CXXFLAGS+=$(OPTS)
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_61,compute_60]

OBJ=batch_knn.o graph_pooling.o group_points.o kernel_correlation.o wrap.o
OBJS=$(addprefix $(OBJDIR), $(OBJ))

all: obj $(SLIB)

$(SLIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cc
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu
	$(NVCC) $(ARCH) $(INCLUDES) --compiler-options "$(NVCCFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(OBJDIR)/*

