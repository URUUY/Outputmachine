This is the UL CSIS 2025 Final Year Project by Tianyu Sun.

Intro:
This project aim to use deeplabv3 and unet to train a segmentation model by dataset FoodSeg103.
In the last steps, using ResNet50 to recognize each parts of segmentation result to evaluate correctly.
Also provide two voting function to improve prediction probability.

folders:
  deeplabv3-plus-pytorch
  deeplabv3-plus-pytorch copy
  unet-pytorch
  unet-pytorch
Those four folders used to train four model, can be found on subfolder trainedmodel in each folders.
ref by https://github.com/bubbliiiing/deeplabv3-plus-pytorch and https://github.com/bubbliiiing/unet-pytorch

FoodSeg104toIRds.ipynb generate FoodSeg104toIR dataset,
then use IRtrainer.ipynb to training Image Recognition model.

environment:
  python 3.9.21
  torch 2.6.0

How to use?

run_predicted.py is the predicted program, user need input the path of food image, then program will give four predicted and save in folder outputimage,
named (original_name)_predicted1~4.
next, user need choose one kind of voting method(majority voting, weighted voting based on miou), input 1 or 2 to choose.
the predicted image will named (original_name)_predicted5.

The ingredents will saved result.txt, and the error correction results using image recognition will be saved in result2.txt, based on the image segmented into parts, in folder IRimage, by timestamp.

next, nutrition_cal.py will use nutritionX api to get the calories (per serving) of the ingredients in result.txt and give the weight of each serving, saved in food_calories_output.csv.

