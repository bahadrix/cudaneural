#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include "parann.cuh";

//Sigmoid function
__device__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

//Derivative of sigmoid function
__device__ float d_sigmoid(float x) {
	return x * (1 - x);
}

__global__ void train(
	float *input, float *hidden, float *output, // node values
	unsigned int inputSize, unsigned int hiddenSize, unsigned int outputSize,  // node counts
	float *weights_i2h, float *weights_h2o,  // weight infos
	float *trainingInput, float *trainingOutput,
	unsigned int epochCount) {

}



__global__ void f1(float *input, float *hidden, float *output, // node values
	unsigned int inputSize, unsigned int hiddenSize, unsigned int outputSize,  // node counts
	float *weights_i2h, float *weights_h2o,  // weight infos
	unsigned char *trainingInput, unsigned char *trainingOutput,
	unsigned int epochCount, unsigned int iteration) {

	int id = blockDim.x  * blockIdx.x + threadIdx.x;
	

	int from = id / hiddenSize;
	int to = id % hiddenSize;

	
	
	
	float value = trainingInput[iteration * inputSize + from] * weights_i2h[inputSize * to + from];

	atomicAdd(&hidden[to], value);

	//input[from] = trainingInput
}

void setupNN2(NN2* nn2);
void randomizeWeights(NN2* nn2);
int trainWithGPU(NN2* nn2, unsigned char *trainingInput, unsigned char *trainingOutput, int epoch);
bool cudaCheck(cudaError_t, char*);



int main() {


	cout << "SETUP PHASE" << LINE;	
	cout << "Setting up training set for " << TRAIN_SIZE << " elements....";
	srand(time(NULL));

	//create  training dataset
	unsigned char inputArray[TRAIN_SIZE * (INPUT_COUNT + 1)];
	unsigned char outputArray[TRAIN_SIZE * (OUTPUT_COUNT)];

	

	for(int t = 0; t < TRAIN_SIZE; t++) {
		
		inputArray[INPUT_COUNT*t + 0] = rand() % 2;			
		inputArray[INPUT_COUNT*t + 1] = rand() % 2;			
		inputArray[INPUT_COUNT*t + 2] = 1; //bias
		
		outputArray[t] = inputArray[INPUT_COUNT*t + 0] TEST_OPERATOR inputArray[INPUT_COUNT*t + 1];
		
	}
	cout << "OK\n" << "Setting up neural network [" << INPUT_COUNT << "i, " << HIDDEN_COUNT << "h]....";
	
	//Setup neural network
	NN2 nn2;
	setupNN2(&nn2);

	//Iterasyon dizileri
	unsigned char inputSet[INPUT_COUNT + 1];
	unsigned char outputSet[OUTPUT_COUNT];
	
	cout << "OK\n";
	

	//initialize the GPU
	//cudaError_t initGPU();
	bool errorExist = 0;
	cout << "Initializing device..";
	errorExist |= cudaCheck(cudaSetDevice(0),"");

	

	trainWithGPU(&nn2,inputArray, outputArray, MAX_EPOCH);


	errorExist |= cudaCheck(cudaDeviceReset(),"Device reset");


	clock_t start = clock();
	cout << LINE << "EXECUTION PHASE" << LINE;
	cout << "Training started [" << MAX_EPOCH << " epoch]...";
	if(!errorExist) {
		cout << "\n Completed with error";
	}
	getchar();
	
	return 0;
}



int trainWithGPU(NN2* nn2, unsigned char *trainingInput, unsigned char *trainingOutput, int epoch) {
	unsigned char *d_trainingInput, *d_trainingOutput;
	float *d_inputArray, *d_hiddenArray, *d_outputArray;
	float *d_weight_i2h, *d_weight_h2o;
	bool errStat = 1;

	cout << "OK\n" << "Allocating memory on GPU..";


	// ALLOCATING MEMORY
	errStat &= cudaCheck( // Allocate input
		cudaMalloc((void**)&d_inputArray, nn2->inputCount * sizeof(float)),
		"Memory allocate error: input");

	errStat &= cudaCheck( // Allocate hidden
		cudaMalloc((void**)&d_hiddenArray, nn2->hiddenCount * sizeof(float)),
		"Memory allocate error: hidden");

	errStat &= cudaCheck( // Allocate output
		cudaMalloc((void**)&d_outputArray, nn2->outputCount * sizeof(float)),
		"Memory allocate error: output");

	errStat &= cudaCheck( // Allocate i2h
		cudaMalloc((void**)&d_weight_i2h, 2 * nn2->inputCount * nn2->hiddenCount * sizeof(float)),
		"Memory allocate error: i2h weights");

	errStat &= cudaCheck( // Allocate h2o
		cudaMalloc((void**)&d_weight_h2o, 2 * nn2->hiddenCount * nn2->outputCount * sizeof(float)),
		"Memory allocate error: h2o weights");

	errStat &= cudaCheck( // Allocate trainingInput
		cudaMalloc((void**)&d_trainingInput, TRAIN_SIZE * nn2->inputCount * sizeof(float)),
		"Memory allocate error: trainingInput");

	errStat &= cudaCheck( // Allocate trainingOutput
		cudaMalloc((void**)&d_trainingOutput, TRAIN_SIZE * nn2->outputCount * sizeof(float)),
		"Memory allocate error: trainingOutput");

	
	// COPY DATA ------------------------------------------------------------------
	//cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	
	cout << "OK\nTransferring structure..";

	errStat |= cudaCheck( // Copy input
		cudaMemcpy(d_inputArray,nn2->input, nn2->inputCount * sizeof(float), cudaMemcpyHostToDevice),
		"Memory allocate error: copying input");

	errStat |= cudaCheck( // Copy hidden
		cudaMemcpy(d_hiddenArray,nn2->hidden, nn2->hiddenCount * sizeof(float), cudaMemcpyHostToDevice),
		"Memory allocate error: copying hidden");

	errStat |= cudaCheck( // Copy output
		cudaMemcpy(d_outputArray,nn2->output, nn2->outputCount * sizeof(float), cudaMemcpyHostToDevice),
		"Memory allocate error: copying input");

	errStat |= cudaCheck( // Copy i2h weights
		cudaMemcpy(d_weight_i2h,nn2->weight_i2h, 2 * nn2->inputCount * nn2->hiddenCount * sizeof(float), cudaMemcpyHostToDevice),
		"Memory allocate error: copying i2hw");

	errStat |= cudaCheck( // Copy h2o weights
		cudaMemcpy(d_weight_h2o,nn2->weight_h2o, 2 * nn2->hiddenCount * nn2->outputCount * sizeof(float), cudaMemcpyHostToDevice),
		"Memory allocate error: copying i2hw");

	cout << "OK\n" << "Transferring training data..";
	
	errStat |= cudaCheck( // Copy output
		cudaMemcpy(d_trainingInput,trainingInput, TRAIN_SIZE * nn2->inputCount * sizeof(unsigned char), cudaMemcpyHostToDevice),
		"Memory allocate error: copying input");
	
	errStat |= cudaCheck( // Copy output
		cudaMemcpy(d_trainingOutput,trainingOutput, TRAIN_SIZE * nn2->outputCount * sizeof(unsigned char), cudaMemcpyHostToDevice),
		"Memory allocate error: copying output");
	
	cout << "OK\n";

	//-----


	int totalWork = TRAIN_SIZE * MAX_EPOCH;
	
	int i2hLinkCount = nn2->inputCount * nn2->hiddenCount;
	int h2oLinkCount = nn2->hiddenCount * nn2->outputCount;
	int it = 0;

	f1<<<totalWork/BLOCK_SIZE, BLOCK_SIZE>>>(
		d_inputArray, d_hiddenArray, d_outputArray,
		nn2->inputCount, nn2->hiddenCount, nn2->outputCount,
		d_weight_i2h, d_weight_h2o,
		d_trainingInput, d_trainingOutput,
		MAX_EPOCH, it);

	errStat |= cudaCheck(cudaGetLastError(), "Kernel execution error");
	errStat |= cudaCheck(cudaDeviceSynchronize(), "Device synchronize error");

	cudaFree(d_inputArray);
	cudaFree(d_hiddenArray);
	cudaFree(d_outputArray);
	cudaFree(d_weight_i2h);
	cudaFree(d_weight_h2o);
	cudaFree(d_trainingInput);
	cudaFree(d_trainingOutput);

	return 0;

}



void setupNN2(NN2* nn2) {

	nn2->inputCount = INPUT_COUNT + 1; 
	nn2->hiddenCount = HIDDEN_COUNT + 1;
	nn2->outputCount = OUTPUT_COUNT;

	//Allocate the memory ***
	nn2->input = (float*)malloc(nn2->inputCount * sizeof(float)); // +1 for bias
	nn2->hidden = (float*)malloc(nn2->hiddenCount * sizeof(float));
	nn2->output = (float*)malloc(nn2->outputCount * sizeof(float));

	// 20 * 8 = 160 Byte'ý bir arada veremeyecekse ne baslarim oyle bellege
	nn2->weight_i2h = (float*)calloc(2 * nn2->hiddenCount * nn2->inputCount, sizeof(float));
	nn2->weight_h2o = (float*)calloc(2 * nn2->outputCount * nn2->hiddenCount, sizeof(float));
	
	//Set activation function
	//nn2->activator = &activator;
	//nn2->delta = &delta;

	cout << "OK\n" << "Randomizing weights..";
	
	//Initialize the weights	
	randomizeWeights(nn2);

}

void randomizeWeights(NN2* nn2) {
	

	for(int i = 0; i < nn2->inputCount; i++) {
		for(int h = 0; h < nn2->hiddenCount; h++) {
			// for accessing second layer: (nn2->inputCount * nn2->hiddenCount * layernum) + nn2->inputCount * h + i
			nn2->weight_i2h[nn2->inputCount * h + i] = (RANDOM_float * 4) - 2;
		}
	}

	for(int h = 0; h < nn2->hiddenCount; h++) {
		for(int o = 0; o < nn2->outputCount; o++) {
			nn2->weight_h2o[nn2->hiddenCount * o + h] = (RANDOM_float * 4) - 2;
		}
	}

}

bool cudaCheck(cudaError_t cudaStatus, char* errorStr) {
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s", errorStr);
		return false;
    }
	return true;
}