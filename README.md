# Intenship-Description-Task-1
As task not contain dataset, isuggest to use kaggle water meter dataset LINK

Ocr taskcosists from two steps:
1) Semantic segmentation model to loacte a numbers of the watrer meter and crop it
2) OCR model to predict exact numbers on the cropped image

Data is not prepared for segmentation due to not same shape, also it's useful to check if there is an empty masks or image id is not the same as mask id.
Seg_train.ipynb contains some visualization of the provided data for each step.
After checking we rezizing it into shape (256, 256) via opencv and as usual we should not forget to use augmentations with albumentations.
Now splititing it into validation and training parts and we are ready to train our model

The results are:

TABLE + IMG

Up next we should prepare dataset for ocr model.
Before starting cropping dataset, should to say, i will use masks from dataset, not output of the segmentation model because it's better to train ocr model on the best dataset. Naturally i will create a prediction program for an single image using mask from segmentation model.

After using bitwise opencv operation on images with help of the masks, we need to care about rotation of the new cropped image:

IMG

After rotating:

IMG

Now it's time to create a new dataset. Because of limited time i will use already created one: tap here to **download** LINK

