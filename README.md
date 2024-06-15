<h1 align='center'>UNetKAN</h1>

### This is a warehouse for UNetKAN-Pytorch-model, can be used to train your medical-image-datasets for segmentation tasks.
### The code mainly come from [official source code](https://github.com/CUHK-AIM-Group/U-KAN)  

### [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)  
### [U-KANMakesStrong Backbone for Medical Image Segmentation and Generation](https://arxiv.org/abs/2406.02918)

![image](https://github.com/CUHK-AIM-Group/U-KAN/blob/main/assets/framework-1.png)  

## Preparation
### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### Download Datasets
[CVC-ClinicDB Dataset](https://www.kaggle.com/datasets/balraj98/cvcclinicdb?select=PNG)  
[Kvasir Dataset](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation?select=kvasir-seg)

## Project Structure
```
├── datasets: Load datasets
    ├── CVC.py: CVC-ClinicDB & Kvasir Datasets
    ├── transforms.py: image data aug methods
├── models: UNetKAN Model
    ├── build_models.py: Construct UNetKAN & UNetMLP models
    ├── kan.py: Define KAN model replaces MLP layers
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── losses.py: DiceLoss
    ├── metrics.py: Define Metrics (pixel_acc, f1score, miou)
    ├── utils.py: Record various indicator information and output and distributed environment
├── engine.py: Function code for a training/validation process
├── estimate_model.py: Visualized performance of pretrained model in validset
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___Kvasir_path___, ___ClinicDB_path___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters.  

## Train this model
### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.fintune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
Using a specified part of the GPUs: for example, I want to use the ___second___ and ___fourth___ GPUs:  
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
For the specific number of GPUs on each machine, modify the value of ___--nproc_per_node___.  
If you want to specify a certain GPU, just add ___CUDA_VISIBLE_DEVICES=___ to specify the index number of the GPU before each command.  
The principle is the same as single-machine multi-GPU training:  
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## ONNX deployment
### step 1: ONNX export
```bash
python onnx_export.py
```

### step2: ONNX optimise
```bash
python onnx_optimise.py
```

### step3: ONNX validate
```bash
python onnx_validate.py
```


## Citation
```
@article{li2024ukan,
  author    = {Chenxin Li and Xinyu Liu and Wuyang Li and Cheng Wang and Hengyu Liu and Yixuan Yuan},
  title     = {U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation},
  journal   = {arXiv preprint arXiv:2406.02918},
  year      = {2024}
}
```

```
@article{liu2024kan,
  title={Kan: Kolmogorov-arnold networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```
