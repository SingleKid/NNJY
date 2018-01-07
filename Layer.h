#pragma once
#include <stdio.h>
enum LayerType {
	Input = 0, Output = 1, Hidden = 2, Unknown = 3,
};
struct Layer {
	double * value;
	double * value2;
	int neural_amount;
	LayerType type; // 0 for input, 1 for output, 2 for hidden

	Layer(LayerType type, int amount);
	void fillAsInput(double * data);
	void fillIn(const double* data);
	void clear();
	void print()
	{
		printf("layer type : %d, length = %d\n value : ", type, neural_amount);
		for (int i = 0; i < neural_amount; i++)
		{
			printf("%.8lf, ", value[i]);
		}
		printf("\n value2: ");
		for (int i = 0; i < neural_amount; i++)
		{
			printf("%.8lf, ", value2[i]);
		}
		printf("\n");
	}
	Layer();
	~Layer();
};

void malloc_weights_for(int front, int back, double **& out, const double ** table = NULL);
double normolizeFunction(double x);