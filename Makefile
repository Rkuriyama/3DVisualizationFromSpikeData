CXXFLAGS= -g -Wall -std=c++11
LDLIBS= -lGL -lglfw

OBJECTS= $(patsubst %.cpp,%.o,$(wildcard *.cpp))
TARGET= sample


${TARGET} : ${OBJECTS}
	${LINK.cc} $^ ${LOADLIBES} ${LDLIBS} -o $@

first_sample: first_sample.o
	nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -o first_sample first_sample.o -L/usr/lib/nvidia-compute-utils-440:amd64 -lGL -lGLU -lglut

first_sample.o: first_sample.cu
	nvcc -ccbin g++ -m64 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_70,code=sm_70 -o first_sample.o -c first_sample.cu -lm

.PHONY : clean
clean :
	-${RM} ${TARGET} ${OBJECTS} *~ .*~ core

