CU_APPS=metrics
# CPP_APPS=test_kernel

all: ${CPP_APPS} ${CU_APPS}

## Todo: try out O3 optimization
%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.hpp
	gcc -O2 -std=c++17 -o $@ $<
clean:
	rm -f ${CU_APPS} ${CPP_APPS}
