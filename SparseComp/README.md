# SparseComp: Sparsity-imposed DNN Compression
SparseComp is a new iterative algorithm that compresses selective layers of a pre-trained DNN by retraining and imposing sparsity constraints simultaneously. In addition, it also employs layer separation to reduce the amount of runtime memory space needed to store layer activations during intermittent inference.

This folder contains all the code needed to run it and is structured as follows:
* `sparsecomp.py`: plug-and-play algorithm to compress DNN.
* `conversion.py`: tool to convert pytorch (compressed) model into C model.
* `decomposition.py` and `VBMF.py`: are used by sparsecomp to decompose layers, code respectively from https://github.com/jacobgil/pytorch-tensor-decompositions and https://github.com/CasvandenBogaard/VBMF.

## SparseComp

To compress a pytorch model all you need is the compress function:

```
from sparsecomp import compress_NN_models
```

Assuming your model is named `model`, you just need to specify the target size in KBs and the various [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html):

```
compress_NN_models(model, 100, train_loader, test_loader, val_loader=val_loader)
```

There are additional optional parameters to adjust the compression and other utility functions explained in the file comments.

## Conversion

The conversion to c is performed via these steps:
1. Unrolling the model: all the Sequential layers are analyzed recursively, creating an ordered list of all the different layers of the model. This list is used as a forward function by applying each layer to the input which will gradually change shape arriving at the final output.
This method was used to allow each layer to have the details of the output
dimensions, necessary for execution in C.
2. For each element of the list:
    1. the type of layer is analysed,
    2. applied to the output of the previous layer,
    3. details of the layer are concatenated to the different strings: declaration and definition of parameters (model.c),
    4. the layers that contain stored parameters are printed following the correct representation in the corresponding files (e.g. fc1.h),
3. Once all the layers have been analyzed, the model.c file is written using the stored strings (2.c), the input used is saved in input.h, and all the files that require to be imported (e.g. fc1.h) are written in the model.h file.

Note: This method however has problems with models whose forward function does
not correspond to the list of layers so modifications may be necessary. We are exploring other options with `torch.fx` to infer the model structure into an intermediate representation and then convert it into C.

### Running the conversion

You need to import only the conversion function:

```
from conversion import save_compressed_model
```

Then specify the matrix representation (currently only 'csr' supported) and an input from the dataset:

```
save_compressed_model(model, 'csr', input_data=single_sample)
```

Again there are more optional paramters to control the process.

The result will be a directory containing the C model:
```
model_directory
├── model.c
├── model.h
└── headers
    ├── conv1.h
    ├── ...
    ├── fc1.h
    ├── ...
    ├── headers.h
    └── input.h (if specified)
```

If we want to convert instead an Early Exit Model then we have to pass both the baseline and the early exit model (along with the exit simulation function):
```
from conversion import save_compressed_early_exit_model
save_compressed_early_exit_model(baseline, exit_model, 'csr', simulate_exit=simulate_exit, input_data=single_sample)
```
The result will be a similar directory containing both models:
```
model_directory
├── model.c
├── model.h
├── model_ee.c
├── model_ee.h
└── headers
    ├── conv1.h
    ├── ...
    ├── fc1.h
    ├── fc1_ee.h
    ├── ...
    ├── headers.h
    ├── headers_ee.h
    └── input.h (if specified)
```

## Code Examples
We will soon upload some Jupyter Notebooks as additional examples of the algorithm on different datasets.
