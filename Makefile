CC=/usr/local/cuda/bin/nvcc
build/gpu/loopy_gpu.o: src/loopy_gpu.cu
	$(CC) -g -I /usr/include/python3.5  --compiler-options '-fPIC -mavx -mavx2' --device-c src/loopy_gpu.cu -o build/gpu/loopy_gpu.o
	$(CC) -g -L/usr/local/cuda/lib64  --compiler-options '-fPIC' -dlink -o build/gpu/loopy_gpu_link.o build/gpu/loopy_gpu.o -lcudadevrt -lcudart
#$(CC) -arch=sm_61 -Xcompiler -fPIC -dlink -shared src/loopy_gpu.cu -I /usr/include/python3.5  -o build/gpu/libloopygpu.so  -L/usr/local/cuda/lib64 -lcudart -lcudadevrt


#	/usr/local/cuda/bin/nvcc -I /usr/include/python3.5 -rdc=true -c src/loopy_gpu.cu -o build/gpu/temp.o
#	/usr/local/cuda/bin/nvcc -dlink -o build/gpu/loopy_gpu.o build/gpu/temp.o -L/usr/local/cuda/lib -lcudart
#
#	ar cru build/gpu/libloopy_gpu.a build/gpu/loopy_gpu.o build/gpu/temp.o
#	ranlib build/gpu/libloopy_gpu.a


#$(CC) --compiler-options '-fPIC' -L/usr/local/cuda/lib -lcudart --device-link build/gpu/loopy_gpu.o --output-file build/gpu/loopy_gpu_dlink.o

clean:
	rm build/gpu/*
