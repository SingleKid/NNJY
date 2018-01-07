#include "RNNJY.h"





RNN::RNN(
	int input,
	int hidden_layer,
	int * hiddens,
	int output,
	int sequence_limitation,
	const double * biases,
	const double *** table
)
{
	time = 0;
	study_rate = 0.1;
	sequence_max_length = sequence_limitation;
	input_neural_amount = input;
	output_neural_amount = output;
	hidden_neural_amount = new int[hidden_layer];
	memcpy(hidden_neural_amount, hiddens, sizeof(int) * hidden_layer);
	hidden_layer_amount = hidden_layer;
	weight_layer_amount = hidden_layer + 1;
	this->biases = new double[weight_layer_amount];
	if (biases) memcpy(this->biases, biases, sizeof(double) * weight_layer_amount);
	else memset(this->biases, 0, sizeof(double) * weight_layer_amount);

	weights = new double**[weight_layer_amount];
	weights2 = new double**[weight_layer_amount];
	tweights = new double**[weight_layer_amount];
	tweights2 = new double**[weight_layer_amount];

	if (!table)
	{
		table = new const double**[weight_layer_amount + hidden_layer];
		memset(table, 0, sizeof(double**)*(weight_layer_amount + hidden_layer));
	}
	malloc_weights_for(input_neural_amount, hiddens[0], weights[0], table[0]);
	malloc_weights_for(input_neural_amount, hiddens[0], weights2[0], table[0]);
	for (int i = 0; i < hidden_layer - 1; i++)
	{
		malloc_weights_for(hiddens[i], hiddens[i + 1], weights[i + 1], table[i + 1]);
		malloc_weights_for(hiddens[i], hiddens[i + 1], weights2[i + 1], table[i + 1]);
		malloc_weights_for(hiddens[i], hiddens[i], tweights[i], table[i + 1 + weight_layer_amount]);
		malloc_weights_for(hiddens[i], hiddens[i], tweights2[i], table[i + 1 + weight_layer_amount]);
	}
	malloc_weights_for(hiddens[hidden_layer - 1], output_neural_amount, weights[hidden_layer], table[hidden_layer]);
	malloc_weights_for(hiddens[hidden_layer - 1], output_neural_amount, weights2[hidden_layer], table[hidden_layer]);
	malloc_weights_for(hiddens[hidden_layer - 1], hiddens[hidden_layer - 1], tweights[hidden_layer - 1], table[weight_layer_amount + hidden_layer - 1]);
	malloc_weights_for(hiddens[hidden_layer - 1], hiddens[hidden_layer - 1], tweights2[hidden_layer - 1], table[weight_layer_amount + hidden_layer - 1]);

	hidden_layers = new Layer*[sequence_limitation];
	input_layers = new Layer*[sequence_limitation];
	output_layers = new Layer*[sequence_limitation];
	for (int j = 0; j < sequence_limitation; j++) {
		hidden_layers[j] = new Layer[hidden_layer];
		input_layers[j] = new Layer(Input, input_neural_amount);
		output_layers[j] = new Layer(Output, output_neural_amount);
		for (int i = 0; i < hidden_layer; i++)
		{
			hidden_layers[j][i] = Layer(Hidden, hiddens[i]);
		}
	}
	sigmas = new double[output_neural_amount];
	memset(sigmas, 0, sizeof(double) * output_neural_amount);

	log_weights = fopen("log_weights.txt", "w");
	log_errors = fopen("log_errors.txt", "w");
}
void RNN::train(
	const double *** inputs,
	const int * sequence_length,
	const double *** expected_outputs,
	int dataset_amount
)
{
	double error = 0.0;
	for (int j = 0; j < 10000; j++)
	{
		for (int i = 0; i < dataset_amount; i++)
		{
			for (time = 0; time < sequence_length[i]; time++)
			{
				singleRun(inputs[i][time]);
				error += getError(expected_outputs[i][time]);
			}
			for (time = sequence_length[i] - 1; time >= 0; time--)
			{
				calculateSigmas(expected_outputs[i][time]);
				lastWeightsCorrection();
				backRun(time == sequence_length[i] - 1);
			}

			error = 0;
		}
	}
}
void RNN::saveWeights()
{
	for (int i = 0; i < input_neural_amount; i++)
	{
		for (int j = 0; j < hidden_neural_amount[0]; j++)
		{
			fprintf(log_weights, "%.2lf,", weights[0][i][j]);
		}
	}
	for (int i = 1; i < hidden_layer_amount; i++)
	{
		for (int j = 0; j < hidden_neural_amount[i]; j++)
		{
			for (int k = 0; k < hidden_neural_amount[i + 1]; k++)
			{
				fprintf(log_weights, "%.2lf,", weights[i][j][k]);
			}
		}
	}
	for (int i = 0; i < lastHidden()->neural_amount; i++)
	{
		for (int j = 0; j < output_neural_amount; j++)
		{
			fprintf(log_weights, "%.2lf,", weights[hidden_layer_amount][i][j]);
		}
	}
	fprintf(log_weights, "\n");
}
RNN::~RNN() {
	for (int i = 0; i < weight_layer_amount; i++)
	{
		delete[] weights[i];
		delete[] weights2[i];
	}
	delete[] weights;
	delete[] weights2;

	delete[] sigmas;


	for (int j = 0; j < sequence_max_length; j++) {
		input_layers[j]->~Layer();
		output_layers[j]->~Layer();
		for (int i = 0; i < hidden_layer_amount; i++)
		{
			hidden_layers[j][i].~Layer();
		}
		delete[] hidden_layers[j];
	}

	delete input_layers;
	delete output_layers;
	delete[] hidden_layers;
}

void RNN::changeWeights2AndWeights()
{
	double *** temp = weights2;
	weights2 = weights;
	weights = temp;

	temp = tweights2;
	tweights2 = tweights;
	tweights = temp;
}
Layer * RNN::lastHidden()
{
	return hidden_layers[time] + hidden_layer_amount - 1;
}
void RNN::lastWeightsCorrection()
{
	Layer * last = lastHidden();
	for (int i = 0; i < last->neural_amount; i++)
	{
		for (int j = 0; j < output_layers[time]->neural_amount; j++)
		{
			weights[hidden_layer_amount][i][j]
				= weights[hidden_layer_amount][i][j] + study_rate * sigmas[j] * last->value2[i];
			//weights[hidden_layer_amount][i][j]
			//	= weights[hidden_layer_amount][i][j] + study_rate * sigmas[j] * last->value2[i];
		}
	}
	//double *** temp = weights;
	//weights = weights2;
	//weights2 = temp;
}
void RNN::backRun(bool is_last)
{
	output_layers[time]->fillAsInput(sigmas);
	if (!is_last)
		for (int j = 0; j < lastHidden()->neural_amount; j++)
			for (int k = 0; k < lastHidden()->neural_amount; k++)
			{
				tweights[hidden_layer_amount - 1][j][k] =
					tweights[hidden_layer_amount - 1][j][k] +
					study_rate * lastHidden()->value2[j] * hidden_layers[time + 1][hidden_layer_amount - 1].value2[k];
			}
	executeLayer(
		output_layers[time],
		lastHidden(),
		weights[hidden_layer_amount],
		is_last ? NULL : hidden_layers[time + 1] + hidden_layer_amount - 1,
		is_last ? NULL : tweights[hidden_layer_amount - 1],
		0,
		true
	);

	for (int i = hidden_layer_amount - 2; i >= 0; i--)
	{
		for (int j = 0; j < hidden_neural_amount[i + 1]; j++)
		{
			for (int k = 0; k < hidden_neural_amount[i]; k++)
			{
				weights[i + 1][k][j] =
					weights[i + 1][k][j] + study_rate * hidden_layers[time][i + 1].value2[j] * hidden_layers[time][i].value2[k];
			}
			if (!is_last) for (int k = 0; k < hidden_neural_amount[i + 1]; k++)
			{
				tweights[i][j][k] =
					tweights[i][j][k] + study_rate * hidden_layers[time][i].value2[j] * hidden_layers[time + 1][i].value2[k];
			}
		}
		executeLayer(
			hidden_layers[time] + i + 1,
			hidden_layers[time] + i,
			weights[i + 1],
			is_last ? NULL : hidden_layers[time + 1] + i + 1,
			is_last ? NULL : tweights[i],
			0,
			true
		);
	}
	for (int j = 0; j < input_layers[time]->neural_amount; j++) {
		for (int k = 0; k < hidden_layers[time][0].neural_amount; k++)
		{
			weights[0][j][k] =
				weights[0][j][k] + study_rate * hidden_layers[time][0].value2[k] * input_layers[time]->value2[j];
		}
	}
	//executeLayer(hidden_layers, input_layer, weights[0], 0, true);

}
void RNN::calculateSigmas(const double * targets)
{
	double out = 0;
	double target = 0;
	double ob = 0;
	for (int i = 0; i < output_neural_amount; i++)
	{
		out = output_layers[time]->value2[i];
		target = targets[i];
		sigmas[i] = (target - out) * out * (1 - out);
	}
}
void RNN::executeLayer(
	Layer * front,
	Layer * back,
	double ** weight,
	Layer * additional_layer,
	double ** tweight,
	double bias,
	bool inverse
)
{
	if (!inverse)
	{
		back->clear();
		for (int i = 0; i < back->neural_amount; i++)
		{
			for (int j = 0; j < front->neural_amount; j++)
			{
				back->value[i] += (front->value2[j] * weight[j][i]);
			}
			if (additional_layer) for (int j = 0; j < back->neural_amount; j++)
			{
				back->value[i] += additional_layer->value2[j] * tweight[j][i];
			}
		}
		for (int i = 0; i < back->neural_amount; i++)
		{
			back->value2[i] = normolizeFunction(back->value[i]);
		}
		//back->same_check();
		//printf("go front\t");
		//back->print();
	}
	else {
		for (int i = 0; i < back->neural_amount; i++)
		{
			back->value[i] = back->value2[i] * (1 - back->value2[i]);
			back->value2[i] = 0;
			for (int j = 0; j < front->neural_amount; j++)
			{
				back->value2[i] += front->value2[j] * weight[i][j];
			}
			if (additional_layer) for (int j = 0; j < back->neural_amount; j++)
			{
				back->value2[i] += additional_layer->value2[j] * tweight[i][j];
			}
			back->value2[i] *= back->value[i];
		}
		//printf("go back\t");
		//back->print();
	}
}
void RNN::singleRun(const double * input)
{
	input_layers[time]->fillIn(input);
	executeLayer(
		input_layers[time],
		hidden_layers[time],
		weights[0],
		time == 0 ? NULL : hidden_layers[time - 1],
		time == 0 ? NULL : tweights[0],
		biases[0]
	);
	for (int i = 0; i < hidden_layer_amount - 1; i++)
	{
		executeLayer(
			hidden_layers[time] + i,
			hidden_layers[time] + i + 1,
			weights[i + 1],
			time == 0 ? NULL : hidden_layers[time - 1] + i + 1,
			time == 0 ? NULL : tweights[i + 1],
			biases[i + 1]
		);
	}
	executeLayer(
		hidden_layers[time] + hidden_layer_amount - 1,
		output_layers[time],
		weights[hidden_layer_amount],
		NULL,
		NULL,
		biases[hidden_layer_amount]
	);
}
