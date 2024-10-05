CU_APPS=metrics
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
