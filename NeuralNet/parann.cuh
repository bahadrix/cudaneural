#ifndef PARALLELANN_H
#define PARALLELANN_H


#define INPUT_COUNT 2
#define HIDDEN_COUNT 4
#define OUTPUT_COUNT 1
#define BLOCK_SIZE 256
#define MAX_EPOCH 512
#define RESULT_PER_EPOCH 100
#define LEARNING_RATE 0.9
#define TRAIN_SIZE 100
#define ALPHA 0.9
#define TEST_OPERATOR ^
//disable randomizing for testing
#define RANDOM_FLOAT 0.5 
//#define RANDOM_FLOAT ((float)rand()/RAND_MAX)
#define LINE "\n-------------------------------------------------------------------------------\n"

using namespace std;


// 2-layer neural network
typedef struct NN2 {
	unsigned int inputCount; 
	unsigned int hiddenCount;
	unsigned int outputCount;
	
	float* input;
	float* hidden;
	float* output;

	float* weight_i2h; // input to hidden layer weights
	float* weight_h2o; // hidden to output weights;

} NN2;

#endif