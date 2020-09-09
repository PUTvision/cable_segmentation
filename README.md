## Deep neural network cable segmenter - part of REMODEL EU project

Built with:
* PyTorch
* Pytorch Ligtning
* OpenCV

The names of the files are pretty self-explanatory:
* `annotate.py` - creates binary masks for the images in the `data/img` folder (paint left click - foreground, right click - background, middle button - segments the image with GrabCut to create the mask). Masks are saved to `data/msk`.
* `switch_bg.py` - creates extra images by replacing the background of all images in the `data` folder based on the masks created earlier using all the images in `data/bg` folder. Resulting images are saved to `dataset` folder.
* `segment.py` - creates the deep learning model, data loaders and trains the neural network based on the images and masks in the `dataset` folder.
* `inference.py` - reads a checkpoint and performs prediction on a single image.

In order to run the project, you need to download the following assets and unzip them into the project folder:

-->>>[*CLICK*](https://chmura.put.poznan.pl/s/gLrLzuiXk7YNPuv/download "data download link")<<<--

The assets are the complete image, mask and background set (the `data/` folder), the dataset prepared for training (the `dataset/` foler) and the neural network checkpoint (the `best_model.ckpt` file)
