# pcb_def_detection
Learning how to using the YOLOv8 to make instance segmentation for PCB defect detection

# [Test Env.]
### SDKManager
- Device: Jetson AGX Orin 32G
- L4T: 36.4.3
- Jetpack: 6.2
- CUDA: 12.6.68
- cuDNN: 9.3.0.75
- Python: 3.10.12
### Python packages
- torch                 2.5.0
- torchaudio            2.5.0
- torchvision           0.20.0
- ultralytics           8.3.77
- ultralytics-thop      2.0.14
- numpy                 1.23.5
# [Requred packages]
### pip3
```
sudo apt install python3-pip
```
### ultralytics
```
pip install ultralytics
```

# [Refer Doc.]
https://docs.ultralytics.com/quickstart/#install-ultralytics
https://www.jetson-ai-lab.com/tutorial_ultralytics.html

# [Issue record]
### pip install ultralytics <br/> bash: pip: command not found
```
sudo apt install python3-pip
```
### A module that was compiled using NumPy 1.x cannot be run in <br/> NumPy 2.1.1 as it may crash. To support both 1.x and 2.x <br/> versions of NumPy, modules must be compiled with NumPy 2.0. <br/> Some module may need to rebuild instead e.g. with 'pybind11>=2.12'. <br/><br/> If you are a user of the module, the easiest solution will be to <br/> downgrade to 'numpy<2' or try to upgrade the affected module. <br/> We expect that some modules will need time to support NumPy 2.
```
pip uninstall numpy
```
```
pip install numpy==1.23.5
```
### ERROR: Project file:///home/arbor/Downloads/ultralytics has a 'pyproject.toml' and its build backend is missing the 'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it cannot be installed in editable mode. Consider using a build backend that supports PEP 660.
```
pip install -U pip
```
```
sudo reboot
```
### torch.cuda.is_available(): False <br/> torch.cuda.device_count(): 0 <br/> os.environ['CUDA_VISIBLE_DEVICES']: None <br/> See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.
- https://pypi.jetson-ai-lab.dev/
- Ex: CUDA 12.6(cmd: nvcc -V), https://pypi.jetson-ai-lab.dev/jp6/cu126
### if the https://pypi.jetson-ai-lab.dev/ has been dead, please change to https://pypi.jetson-ai-lab.io/
- https://pypi.jetson-ai-lab.io/
- Ex: CUDA 12.6(cmd: nvcc -V), https://pypi.jetson-ai-lab.io/jp6/cu126
### if nvcc command not found
```
sudo gedit ~/.bashrc

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
- Nvidia forum: https://forums.developer.nvidia.com/t/help-me-with-correct-pytorch-and-torchvision-versions-requirement-for-jetpack-6-2-1-orin-super/343688/47
```
pip uninstall torch
```
```
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip3 install --force-reinstall --no-cache-dir -U torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```
### Be careful when you install other packages that can override your previous packages.
- I recommend always to do this:
```
export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
```
### Missing cudss link
```
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss
```
### Test cudss
```
$ python3 -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
torch 2.8.0 built for CUDA 12.6
```
### RuntimeError: TensorRT does not currently build wheels for Tegra systems
- cannot running in venv, if wanna running on VM, need make a symbol link

# [Annotation via roboflow]
- Please refer to roboflow_tutorial.docx

# [Train your custom model]
- After you download the dataset(annotated), then you can start to training custome model via below script
- Ex: training.py
```
touch training.py && gedit training.py 
```
```
from ultralytics import YOLO
import torch

# load a  pretrained model (recommended for training)
model = YOLO('yolov8m-seg.pt')

# empty cuda cache, avoid out of memory
torch.cuda.empty_cache()

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640, batch=-1)
```
### Using CLI
```
yolo train data=data.yaml model=yolov8m-seg.pt epochs=100 imgsz=640 batch=-1
```
### If the terminal shows 'Starting training for 100 epochs...', it means the training has started.
```
Transferred 531/537 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /home/arbor/Downloads/train/train/labels.cache... 132 images, 20
val: Scanning /home/arbor/Downloads/train/valid/labels.cache... 26 images, 5 bac
Plotting labels to runs/segment/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001111, momentum=0.9) with parameter groups 86 weight(decay=0.0), 97 weight(decay=0.0005), 96 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/segment/train
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      1/100      4.03G      1.424      2.865      3.675      1.475         12   
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         26         56       0.84      0.175      0.133     0.0873      0.836      0.167      0.128     0.0749

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
      2/100      4.07G       1.01      2.137      2.407      1.237         16   
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         26         56      0.431      0.233      0.118     0.0804      0.292      0.192     0.0783     0.0517
### Training finished.
100 epochs completed in 0.686 hours.
Optimizer stripped from runs/segment/train/weights/last.pt, 54.8MB
Optimizer stripped from runs/segment/train/weights/best.pt, 54.8MB

Validating runs/segment/train/weights/best.pt...
Ultralytics 8.3.77 🚀 Python-3.10.12 torch-2.5.0 CUDA:0 (Orin, 30697MiB)
YOLOv8m-seg summary (fused): 105 layers, 27,225,279 parameters, 0 gradients, 110.0 GFLOPs

                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         26         56      0.909      0.607      0.718      0.571      0.871       0.58      0.692      0.522
             Dry_joint          1          9          1       0.24      0.519      0.401          1      0.233      0.519      0.391
Incorrect_installation         11         30      0.773      0.793      0.855       0.78      0.739      0.755      0.822      0.649
            PCB_damage          1          2      0.864          1      0.995      0.798       0.88          1      0.995      0.847
         Short_circuit         13         15          1      0.395      0.502      0.305      0.865      0.333       0.43        0.2
Speed: 0.7ms preprocess, 49.1ms inference, 0.0ms loss, 4.1ms postprocess per image
Results saved to runs/segment/train
```

# [Tuning the custom model]
### Using python script
```
from ultralytics import YOLO
import torch

# load a  pretrained model (recommended for training)
model = YOLO('yolov8m-seg.pt')

# empty cuda cache, avoid out of memory
torch.cuda.empty_cache()

# tuning
result = model.tune(data="data.yaml", epochs=30, iterations=100, batch=-1, optimizer="AdamW", imgsz=640)
```
### Tuning finished
```
30 epochs completed in 0.072 hours.
Optimizer stripped from runs/segment/train302/weights/last.pt, 54.8MB
Optimizer stripped from runs/segment/train302/weights/best.pt, 54.8MB

Validating runs/segment/train302/weights/best.pt...
Ultralytics 8.3.75 🚀 Python-3.10.12 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 3070, 7753MiB)
YOLOv8m-seg summary (fused): 245 layers, 27,225,279 parameters, 0 gradients, 110.0 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 5/5 [00:00<00:00,  8.87it/s]
                   all         28         58      0.757      0.669      0.755      0.586      0.748      0.661      0.722      0.522
             Dry_joint          1          9       0.81      0.477      0.739      0.424       0.81      0.477      0.639      0.387
Incorrect_installation         11         30      0.739      0.767      0.799      0.641      0.706      0.733      0.764      0.573
         Short_circuit         13         15      0.683      0.432      0.486      0.349      0.683      0.433      0.489      0.267
           jumper wire          3          4      0.795          1      0.995       0.93      0.795          1      0.995       0.86
Speed: 0.8ms preprocess, 8.9ms inference, 0.0ms loss, 1.9ms postprocess per image
Results saved to runs/segment/train302
💡 Learn more at https://docs.ultralytics.com/modes/train
Saved runs/segment/tune/tune_scatter_plots.png
Saved runs/segment/tune/tune_fitness.png

Tuner: 300/300 iterations complete ✅ (93721.10s)
Tuner: Results saved to runs/segment/tune
Tuner: Best fitness=1.23044 observed at iteration 162
Tuner: Best fitness metrics are {'metrics/precision(B)': 0.8836, 'metrics/recall(B)': 0.68847, 'metrics/mAP50(B)': 0.7632, 'metrics/mAP50-95(B)': 0.62848, 'metrics/precision(M)': 0.87567, 'metrics/recall(M)': 0.68056, 'metrics/mAP50(M)': 0.7469, 'metrics/mAP50-95(M)': 0.57089, 'val/box_loss': 1.68806, 'val/seg_loss': 4.67301, 'val/cls_loss': 0.86198, 'val/dfl_loss': 1.03493, 'fitness': 1.23044}
Tuner: Best fitness model is runs/segment/train164
Tuner: Best fitness hyperparameters are printed below.

Printing 'runs/segment/tune/best_hyperparameters.yaml'

lr0: 0.0008
lrf: 0.00512
momentum: 0.87472
weight_decay: 0.00041
warmup_epochs: 1.3538
warmup_momentum: 0.46736
box: 14.47981
cls: 0.33447
dfl: 1.2533
hsv_h: 0.01629
hsv_s: 0.36925
hsv_v: 0.40902
degrees: 0.0
translate: 0.11759
scale: 0.41567
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.82305
bgr: 0.0
mosaic: 0.90467
mixup: 0.0
copy_paste: 0.0
```
### You can find out the best model in log, like below descript, which mean the train164 is the best one during whole process.
- Tuner: Best fitness model is runs/segment/train164

# [Convert the pytorch model to TensorRT engine]
### Using CLI
```
yolo export model=best.pt format=engine
```
### Using python script
```
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Load a model
model.export(format="engine")
```
# [Execute the python script with TensorRT engine]
### please refet to pcb_defect_test.py
