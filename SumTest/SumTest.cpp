// SumTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

using namespace std;


void printArray(int* input, int size);
int sumStandart(int* input, int size) ;

int _tmain(int argc, _TCHAR* argv[])
{
	const int inputSize = 6;
	
	int input[inputSize];
	int output[inputSize];

	for(int i=0; i < inputSize; i++) {
		input[i] = i+1;
	}
	
	cout << "Starting ar: "; printArray(input, inputSize);
	
	cout << "Calculating standart..";
	int r = sumStandart(input, inputSize);
	cout << "OK. Result: " << r;

	cout << "\nCalculating with arrays..";
	int stepSize = 1;
	const int maxStep = ceil(sqrt(inputSize));
	int mid = inputSize;
	int o = 0, s= 0;
	for(int step = 0; step < maxStep; step++) {
		

		//while {
			
			output[o]


			s += 2;
		//}
		cout << "\nIteration " << step << ": ";
		printArray(output, inputSize);
	}
	cout <<"OK\nResult: " << input[0];
	
	if(input[0] == inputSize * (inputSize+1) / 2) {
		cout << " Correcto Mundo!";
	} else {
		cout << " False Result!";
	}
	
	getchar();
	return 0;
}

int sumStandart(int* input, int size) {

	int r= 0;

	for(int i = 0; i < size; i++) {

		r += input[i];
	}

	return r;
}

void printArray(int* input, int size) {

	cout << "[";
	for(int i = 0; i < size; i++) {
		printf("%02d ", input[i]);

	}
	cout << "]"; 

}