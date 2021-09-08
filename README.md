# UDP
UDP: A Universal Digital-to-Physical World Attack Method


**Clone the repository; install dependencies**

```
git clone https://github.com/hdhcy/UDP.git     
pip install requirements.txt
```

**Download the dataset**

- [digital physical dataset](https://drive.google.com/file/d/1MYT2B-E1ISjjS51y1juBG43ZzTa8GJks/view?usp=sharing).

The downloaded dataset should be placed in the data folder.

**Train the UNet style Digital-to-Physical transformation model**

```
python digital_to_physical_transformation.py --dataroot ./data/d2p --save_path ./data_saved/UNet_trans.pth batch_size --batch_size 32 --n_epochs 400 --lr 1e-3 --gpu 0
```
You can also use our trained transformation [model](https://drive.google.com/file/d/1sKJcTk41LwrfWumUfRSGgigYcIdLPqp3/view?usp=sharing). Please put it in the data_saved (may need to create) folder

**UDP attack method**




