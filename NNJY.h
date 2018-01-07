#pragma once

#include "Layer.h"
#include "MLPJY.h"
#include "RNNJY.h"


MLP * loadMLP(char * filename)
{
	FILE * fp = fopen(filename, "r");
	if (!fp)return NULL;
	



}