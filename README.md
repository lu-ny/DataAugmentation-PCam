# biweekly-report-3-sgreenlund3
biweekly-report-3-sgreenlund3 created by GitHub Classroom

For this biweekly report, I explored data augmentation. I decided to use the patch_camelyon dataset, which is comprised on images of lymph node histopathology and binary identifiers indicated whether the tissue has metastatic (cancerous) tissue.

I started with training/fitting a model without augmentation to provide a baseline (DataAugmentation-Baseline). 
I used the layers described in https://medium.com/analytics-vidhya/deep-learning-tutorial-patch-camelyon-data-set-d0da9034550e to build model. The author mentioned around 73% accuracy. So, I aim to hit that in this first notebook and beat it in my subsequent notebook. 

I then built out a simple augmentation using tensorflow. Since these are medical images with standardization, I chose not to manipulate color very much. I wanted to incorporate the jpeg noise, but need to rework that line since it only takes 1 image at a time. 

Next, I wanted to incorporate albumentation, which is a library that has more robust augmentation tools, and use it in a randAugment model. I was having a lot of trouble with the PCam dataset and tensorflow, so I ended up switching to torch and CIFAR10. I had to move my preprocessing function to a py file and import it as I was running into multiprocessing errors. This notebook isn't fully complete - I'm still running into issues with memory and a few other bugs
 
*NOTE* I uploaded notebooks with only 1 epoch showing, as they take a while to run. I will upload them with the additional epochs as they finish. 
