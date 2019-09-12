#pragma once
#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "Layer.h"

struct RNN {
	int sequence_max_length;
	int hidden_layer_amount;
	int input_neural_amount;
	int output_neural_amount;
	int weight_layer_amount;
	int * hidden_neural_amount;

	Layer ** input_layers;
	Layer ** output_layers;
	Layer ** hidden_layers;

	double *** weights;
	double *** weights2;

	double *** tweights;
	double *** tweights2;

	double * sigmas;
	double * biases;

	FILE * log_weights;
	FILE * log_errors;

	double study_rate;
	RNN() = default;
	RNN(
		int input,
		int hidden_layer,
		int * hiddens,
		int output,
		int sequence_limitation,
		const double * biases = NULL,
		const double *** table = NULL
	);
	void train(
		double *** inputs,
		int * sequence_length,
		double *** expected_outputs,
		int dataset_amount
	);
	~RNN();
	double getError(const double * targets)
	{
		double tot = 0.0;
		for (int i = 0; i < output_neural_amount; i++)
		{
			tot += fabs(output_layers[time]->value2[i] - targets[i]);
		}
		return tot;
	}
	void work(double ** input, int sequence_length, double ** out)
	{
		for (time = 0; time < sequence_length; time++)
		{
			singleRun(input[time]);
			for (int i = 0; i < output_neural_amount; i++)
				out[time][i] = output_layers[time]->value2[i];
		}
	}
private:
	int time;
	void saveWeights();
	void changeWeights2AndWeights();
	Layer * lastHidden();
	void lastWeightsCorrection();
	void backRun(bool is_last);
	void calculateSigmas(const double * targets);
	void executeLayer(
		Layer * front,
		Layer * back,
		double ** weight,
		Layer * additional_layer,
		double ** tweight,
		double bias,
		bool inverse = NULL
	);
	void singleRun(const double * input);
};