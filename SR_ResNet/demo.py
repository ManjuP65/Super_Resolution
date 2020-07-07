import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="/content/drive/My Drive/cs231n/project/Test_Project/checkpoint/model_epoch_180.pth", type=str, help="model path")
parser.add_argument("--image", default="Image_5093.tif", type=str, help="image name")
parser.add_argument("--dataset", default="rocks", type=str, help="dataset name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
            

im_gt = sio.loadmat("/content/drive/My Drive/cs231n/project/Test_Project/testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_gt']
im_b = sio.loadmat("/content/drive/My Drive/cs231n/project/Test_Project/testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_b']
im_l = sio.loadmat("/content/drive/My Drive/cs231n/project/Test_Project/testsets/" + opt.dataset + "/" + opt.image + ".mat")['im_l']


im_input = im_l.reshape(1,1,im_l.shape[0],im_l.shape[1])
im_input = Variable(torch.from_numpy(im_input/255.).float())


model = torch.load(opt.model)["model"]

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()



start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h = out.data[0].numpy().astype(np.float32)


im_h = im_h*255.
im_h[im_h<0] = 0
im_h[im_h>255.] = 255.  
print(im_h)          
im_h = im_h.reshape(im_h.shape[1],im_h.shape[2])

print("Dataset=",opt.dataset)
print("Scale=",opt.scale)
print("It takes {}s for processing".format(elapsed_time))
print("PSNR is",PSNR(im_h, im_gt, shave_border=0))
print("Bicubic PSNR is",PSNR(im_b, im_gt, shave_border=0))
print(im_gt.shape)
print(im_b.shape)
print(im_l.shape)

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt,cmap='gray', vmin=0, vmax=255)
ax.set_title("Ground Truth")

ax = plt.subplot("132")
ax.imshow(im_b,cmap='gray', vmin=0, vmax=255)
ax.set_title("Bicubic")

ax = plt.subplot("133")
ax.imshow(im_h.astype(np.uint8),cmap='gray', vmin=0, vmax=255)
ax.set_title("Prediction")
plt.show()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.savefig("test.png")


