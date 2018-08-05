# dehaze-cGAN
Run in Torch7 platform.

Because the trained models may be produce some different effects in varying platforms and computers, I strongly suggest that you can adopt our trained model when restoring real world haze images.

Testing:

DATA_ROOT=./datasets phase=test_syn th test.lua

Training:

DATA_ROOT=./datasets th train.lua


