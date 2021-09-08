# UDP
UDP: A Universal Digital-to-Physical World Adversarial Attack Method


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

**Obtain correctly classified simulated physical images**

```
python get_correct_images.py --model_trans_path ./data_saved/UNet_trans.pth --dataroot ./data/d2p/test --save_dir ./data/predict_right_images
```

Here the physical images simulated by the trained transformation model are judged and the correctly classified images are retained for the next step of the attack.


**UDP attack method**

```
python UDP_attack.py --model_trans_path ./data_saved/UNet_trans.pth --dataroot ./data/predict_right_images/digital --res_save_path ./data_saved/UDP_res --adv_label 521 --epsilon 16  --num_iter 15  --data_size 4000 --data_batch_size 32 --gpu 0
```

**Can the attack be successful**

After the UDP method, we obtain the corresponding adversarial samples, then print and captur them, and finally cropped them to get the adversarial examples in the physical world.

You can directly modify the adversarial sample path in `show_and_get_predicted.py` to get its adversarial strength.
