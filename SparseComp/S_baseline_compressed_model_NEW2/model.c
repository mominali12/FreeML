#include "ml.h"
#include "headers/headers.h" // header files which contain weights and bias

#define NUM_MODEL_LAYERS 11

__fram sequentialModel Model[NUM_MODEL_LAYERS];

/* Convolutional layer 1 parameters */
__fram tensor weights_Convol_2D_1 = {
	.shape = {CONV1_NNZ_LEN},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv1_w_data,
	.sparse = {
		.offset = &conv1_w_offset,
		.size = &conv1_w_sizes,
		.dimension = 4,
		.shape = {64, 3, 3, 3},
	},
};

__fram tensor bias_Convol_2D_1 = {
	.shape = {64},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv1_b,
};

__fram conv_t param_Conv2D_1 = {
	.stride = {1, 1, 1},
	.padding = PAD_NONE,
	.weights = &weights_Convol_2D_1,
	.bias = &bias_Convol_2D_1,
	.outputDimension = 3,
	.outputShape = {64, 30, 30},
	.type = NORMAL,
	.activation = ACT_NONE,
};

/* Pooling layer 1 parameters */
__fram pooling_t param_maxPool_1 = {
	.type = MAX,
	.size = {2, 2},
	.stride = {2, 2},
	.outputShape = {64, 15, 15},
	.outputDimension = 3,
};

/* Convolutional layer 2 parameters */
__fram tensor weights_Convol_2D_2 = {
	.shape = {CONV2_NNZ_LEN},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv2_w_data,
	.sparse = {
		.offset = &conv2_w_offset,
		.size = &conv2_w_sizes,
		.dimension = 4,
		.shape = {128, 64, 3, 3},
	},
};

__fram tensor bias_Convol_2D_2 = {
	.shape = {128},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv2_b,
};

__fram conv_t param_Conv2D_2 = {
	.stride = {1, 1, 1},
	.padding = PAD_NONE,
	.weights = &weights_Convol_2D_2,
	.bias = &bias_Convol_2D_2,
	.outputDimension = 3,
	.outputShape = {128, 13, 13},
	.type = NORMAL,
	.activation = ACT_NONE,
};

/* Pooling layer 2 parameters */
__fram pooling_t param_maxPool_2 = {
	.type = MAX,
	.size = {2, 2},
	.stride = {2, 2},
	.outputShape = {128, 6, 6},
	.outputDimension = 3,
};

/* Convolutional layer 3 parameters */
__fram tensor weights_Convol_2D_3 = {
	.shape = {CONV3_NNZ_LEN},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv3_w_data,
	.sparse = {
		.offset = &conv3_w_offset,
		.size = &conv3_w_sizes,
		.dimension = 4,
		.shape = {64, 128, 3, 3},
	},
};

__fram tensor bias_Convol_2D_3 = {
	.shape = {64},
	.dimension = 1,
	.strides = {1},
	.dataAddr = &conv3_b,
};

__fram conv_t param_Conv2D_3 = {
	.stride = {1, 1, 1},
	.padding = PAD_NONE,
	.weights = &weights_Convol_2D_3,
	.bias = &bias_Convol_2D_3,
	.outputDimension = 3,
	.outputShape = {64, 4, 4},
	.type = NORMAL,
	.activation = ACT_NONE,
};

/* Pooling layer 3 parameters */
__fram pooling_t param_maxPool_3 = {
	.type = MAX,
	.size = {2, 2},
	.stride = {2, 2},
	.outputShape = {64, 2, 2},
	.outputDimension = 3,
};

/* Flatten layer 1 parameters */
__fram reshape_t param_flatten_1 = {
	.new_shape = {256, 1},
	.new_shape_dim = 2,
};

/* Fully Connected layer 1 parameters */
__fram tensor weights_FC_1 = {
	.shape = {25, 256},
	.dimension = 2,
	.strides = {256, 1},
	.dataAddr = &fc1_w,
};

__fram fully_con_t param_fully_con_1 = {
	.activation = ACT_NONE,
	.sparseness = DENSE,
	.weights = &weights_FC_1,
	.outputShape = {25, 1},
	.outputDimension = 2,
};

/* Fully Connected layer 2 parameters */
__fram tensor weights_FC_2 = {
	.shape = {256, 25},
	.dimension = 2,
	.strides = {25, 1},
	.dataAddr = &fc2_w,
};

__fram tensor bias_FC_2 = {
	.shape = {256, 1},
	.dimension = 2,
	.strides = {1, 1},
	.dataAddr = &fc2_b,
};

__fram fully_con_t param_fully_con_2 = {
	.activation = ACT_NONE,
	.sparseness = DENSE,
	.weights = &weights_FC_2,
	.bias = &bias_FC_2,
	.outputShape = {256, 1},
	.outputDimension = 2,
};

/* Fully Connected layer 3 parameters */
__fram tensor weights_FC_3 = {
	.shape = {64, 256},
	.dimension = 2,
	.strides = {256, 1},
	.dataAddr = &fc3_w,
};

__fram tensor bias_FC_3 = {
	.shape = {64, 1},
	.dimension = 2,
	.strides = {1, 1},
	.dataAddr = &fc3_b,
};

__fram fully_con_t param_fully_con_3 = {
	.activation = ACT_NONE,
	.sparseness = DENSE,
	.weights = &weights_FC_3,
	.bias = &bias_FC_3,
	.outputShape = {64, 1},
	.outputDimension = 2,
};

/* Fully Connected layer 4 parameters */
__fram tensor weights_FC_4 = {
	.shape = {10, 64},
	.dimension = 2,
	.strides = {64, 1},
	.dataAddr = &fc4_w,
};

__fram tensor bias_FC_4 = {
	.shape = {10, 1},
	.dimension = 2,
	.strides = {1, 1},
	.dataAddr = &fc4_b,
};

__fram fully_con_t param_fully_con_4 = {
	.activation = ACT_NONE,
	.sparseness = DENSE,
	.weights = &weights_FC_4,
	.bias = &bias_FC_4,
	.outputShape = {10, 1},
	.outputDimension = 2,
};


void loadModel(){

	// Conv1 layer
	Model[0].type = CONV2D;
	Model[0].ptr = (void *)&param_Conv2D_1;
	// Max Pool: 1
	Model[1].type = MAXPOOL;
	Model[1].ptr = (void *)&param_maxPool_1;
	// Conv2 layer
	Model[2].type = CONV2D;
	Model[2].ptr = (void *)&param_Conv2D_2;
	// Max Pool: 2
	Model[3].type = MAXPOOL;
	Model[3].ptr = (void *)&param_maxPool_2;
	// Conv3 layer
	Model[4].type = CONV2D;
	Model[4].ptr = (void *)&param_Conv2D_3;
	// Max Pool: 3
	Model[5].type = MAXPOOL;
	Model[5].ptr = (void *)&param_maxPool_3;
	// Flatten: 1
	Model[6].type = FLATTEN;
	Model[6].ptr = (void *)&param_flatten_1;
	// FC1 layer
	Model[7].type = FULLYCONNECTED;
	Model[7].ptr = (void *)&param_fully_con_1;
	// FC2 layer
	Model[8].type = FULLYCONNECTED;
	Model[8].ptr = (void *)&param_fully_con_2;
	// FC3 layer
	Model[9].type = FULLYCONNECTED;
	Model[9].ptr = (void *)&param_fully_con_3;
	// FC4 layer
	Model[10].type = FULLYCONNECTED;
	Model[10].ptr = (void *)&param_fully_con_4;

}
