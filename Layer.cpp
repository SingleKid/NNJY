
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Layer.h"
void malloc_weights_for(int front, int back, double **& out, const double ** table)
{
	if (!table) srand(time(0));
	out = new double*[front];
	for (int i = 0; i < front; i++)
	{
		out[i] = new double[back];
		for (int j = 0; j < back; j++)
		{
			if (!table)out[i][j] = ((double)(2.0 * rand()) / ((double)RAND_MAX + 1.0) - 1.0); // make the random value with range[-0.5,0.5] and step 0.01
			else out[i][j] = table[i][j];
		}
	}
}
double normolizeFunction(double x)
{
	return 1.0 / (1 + exp(-x));
}

Layer::Layer(LayerType type, int amount)
{
	this->type = type;
	neural_amount = amount;

	value = new double[amount];
	value2 = new double[amount];
	memset(value, 0, sizeof(double) * amount);
	memset(value2, 0, sizeof(double) * amount);
}
void Layer::fillAsInput(double * data)
{
	memcpy(value2, data, neural_amount * sizeof(double));
}

void Layer::fillIn(const double* data)
{
	if (type == Input)
	{
		memcpy(value2, data, neural_amount * sizeof(double));
	}
	//else {
	//	memcpy(value, data, neural_amount * sizeof(double));
	//	for (int i = 0; i < neural_amount; i++)
	//	{
	//		value2[i] = normolizeFunction(value[i]);
	//	}
	//}
}
void Layer::clear()
{
	memset(value, 0, sizeof(double) * neural_amount);
	if (type != Input)
	{
		memset(value2, 0, sizeof(double) * neural_amount);
	}
}
Layer::Layer() {
	value = NULL;
	neural_amount = 0;
	type = Unknown;
}
Layer::~Layer()
{
	//delete[] value;
	//delete[] value2;
}