build/gpu/loopy_gpu.o: src/loopy_gpu.cu
	/usr/local/cuda/bin/nvcc -I /usr/include/python3.5  --compiler-options '-fPIC' --device-c src/loopy_gpu.cu -o build/gpu/loopy_gpu.o
	/usr/local/cuda/bin/nvcc --compiler-options '-fPIC' -L/usr/local/cuda/lib -lcudart --device-link build/gpu/loopy_gpu.o --output-file build/gpu/loopy_gpu_dlink.o
#	/usr/local/cuda/bin/nvcc -I /usr/include/python3.5 -rdc=true -c src/loopy_gpu.cu -o build/gpu/temp.o
#	/usr/local/cuda/bin/nvcc -dlink -o build/gpu/loopy_gpu.o build/gpu/temp.o -L/usr/local/cuda/lib -lcudart
#
#	ar cru build/gpu/libloopy_gpu.a build/gpu/loopy_gpu.o build/gpu/temp.o
#	ranlib build/gpu/libloopy_gpu.a
