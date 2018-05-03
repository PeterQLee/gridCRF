CC=/usr/local/cuda/bin/nvcc

all: build/gpu/loopy_gpu.o build/gpu/train_gpu.o build/gpu/gpu_link.o 
build/gpu/loopy_gpu.o: src/loopy_gpu.cu 
	$(CC) -g -I /usr/include/python3.5  --compiler-options '-fPIC -mavx -mavx2' --device-c src/loopy_gpu.cu  -o build/gpu/loopy_gpu.o

build/gpu/train_gpu.o: src/train_gpu.cu 
	$(CC) -g -I /usr/include/python3.5  --compiler-options '-fPIC -mavx -mavx2' --device-c src/train_gpu.cu  -o build/gpu/train_gpu.o

build/gpu/gpu_link.o: build/gpu/loopy_gpu.o build/gpu/train_gpu.o
	$(CC) -g -L/usr/local/cuda/lib64  --compiler-options '-fPIC' -dlink -o build/gpu/gpu_link.o build/gpu/loopy_gpu.o build/gpu/train_gpu.o -lcudadevrt -lcudart


clean:
	rm build/gpu/*
