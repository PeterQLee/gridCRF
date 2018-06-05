CC=/usr/local/cuda/bin/nvcc
#--prec-div=true
all: directories build/gpu/loopy_gpu.o build/gpu/train_gpu.o build/gpu/gpu_link.o
directories: build/gpu
	mkdir -p build/gpu

build/gpu/loopy_gpu.o: src/loopy_gpu.cu 
	$(CC) -g -I /usr/include/python3.5  --compiler-options '-fPIC' --device-c src/loopy_gpu.cu  -o build/gpu/loopy_gpu.o

build/gpu/train_gpu.o: src/train_gpu.cu 
	$(CC) -g -I /usr/include/python3.5  --compiler-options '-fPIC' --device-c src/train_gpu.cu  -o build/gpu/train_gpu.o

build/gpu/gpu_link.o: build/gpu/loopy_gpu.o build/gpu/train_gpu.o
	$(CC) -g -L/usr/local/cuda/lib64  --compiler-options '-fPIC' -dlink -o build/gpu/gpu_link.o build/gpu/loopy_gpu.o build/gpu/train_gpu.o -lcudadevrt -lcudart


clean:
	rm build/gpu/*
