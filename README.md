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
```
pip uninstall pytorch
```
```
pip install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```
