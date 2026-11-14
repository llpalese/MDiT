# MDiT

This repository allows you to reproduce the results of the manuscript xxx.xxx. In addition to the necessary scripts, a mini-dataset is provided for testing. Obviously, the datasets used to obtain the manuscript's results cannot be reported as such, and this prevents you from obtaining *exactly* the same results. Furthermore, the dcd files in the dataset are not the original raw files used for the analysis: only the protein is reported after alignment to the initial frame. For reference, the initial frame of each MD run has also been saved in pdb format. 

**However, the system is easily portable to any MD dataset.**

The analyses were performed on a Linux system. To use this software, it is recommended to create a conda environment suitable for the machine on which it will be run. The environment must contain:

- python 3.11.14 *(this is the version that has been tested)*
- JAX 
- Keras
- Numpy
- matplotlib
- MDAnalysis
- VMD

If you're using the GPU version of JAX, pay particular attention to the CUDA version. Also pay attention to the Python version, as not all versions (especially the most recent ones) are compatible with JAX.

***Important!*** All scripts are written in Python using Keras and pure JAX. Therefore, to avoid conflicts, it is necessary to run all scripts in an environment where only JAX is present, avoiding coexistence with Tensorflow and/or Pytorch. A pure Python dataloader is in the train.py script, which only requires that the training files are properly formatted and placed in a folder system like the one below (i.e. the training, validation and test files must be separated into different folders).

## How to train the iTransformer
To train the iTransformer, you need to call the train.py file in the appropriate directory. The required arguments are:

	--train_dir (a string indicating the path to the directory containing the training data);
	--val_dir (a string indicating the path to the directory containing the validation data);
	--test_file (a string indicating the path to the directory containing the test file);

All files must be in npz format. The npz format files are matrices containing the Cartesian coordinates of the selected atoms (usually the alpha carbons, as in this case). The folder containing the training, validation, and test data must contain the subfolder for the training files, the subfolder for the validation files, and the folder containing the individual test file. So the scheme is:

```
dataset_folder/
├── train_subfolder/                  
│   ├── your_train_file1.npz
│   ├── your_train_file2.npz
│	...
├── validation_subfolder/
│   ├── your_validation_file1.npz
│   ├── your_validation_file2.npz
│	...
└── test_subfolder/
    └── your_test_file.npz   
```
	                         
The optional arguments are: 

	--lookback (integer, default 60; the number of MD steps that is used as input);
	--forecast (integer, default 50; the number of pseudo-dynamics steps that the Transformer must predict, i.e. the final horizon for Curriculum Learning prediction);
	--initial_forecast (integer, default 10; the starting horizon for Curriculum Learning prediction);
	--forecast_increment (integer, default 20; the forecast increment step per phase for Curriculum Learning);
	--batch_size (integer, default 64; batch dimension);
	--epochs (integer, default 50; epochs for each phase of Curriculum Learning).
	
For example, assuming you are in the directory containing train.py (in the approrpiate Conda environment) to launch training digit:
```
bash
python train.py --train_dir your_path/dataset_folder/train_subfolder --val_dir your_path/dataset_folder/validation_subfolder --test_file your_path/dataset_folder/test_subfolder/your_test_file.npz --initial_forecast 10 --forecast_increment 20 --forecast 100
```



