from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os
import random
import matplotlib.pyplot as plt
import numpy as np


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class VTB_eval:
    def __init__(self, input_dir):
        # Take 5 classes for testing
        self.im_list = os.listdir(input_dir)[: min(5, len(os.listdir(input_dir)))]

class VTB_other:
    def __init__(self, input_dir, gt_dir, num_examples):
        try: 
            self.im_list = os.listdir(input_dir)
        except:
            self.im_list = ['Basketball', 'Biker', 'Freeman4']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            for _ in range(num_examples):
                input0_rand = random.randint(1, len(os.listdir(os.path.join(input_dir, item))) - 2)
                self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + f'/{self.make_basename(input0_rand)}')).unsqueeze(0)))
                self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + f'/{self.make_basename(input0_rand + 2)}')).unsqueeze(0)))
                self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + f'/{self.make_basename(input0_rand + 1)}')).unsqueeze(0)))

    def make_basename(self, index):
        index = str(index)
        try:
            return '0'*(4-len(index)) + index + '.jpg'
        except:
            return '0'*(4-len(index)) + index + '.png'

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        av_psnr = 0
        func = lambda img: np.moveaxis(img.cpu().detach().numpy().squeeze(0), 0, -1)
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        for idx2 in range(len(self.im_list)):
            for idx in range(len(self.input0_list)):
                fig_size = plt.rcParams['figure.figsize']
                fig, ax = plt.subplots(1, 2, figsize=(fig_size[0] * 2, fig_size[1] * 1))
                if not os.path.exists(output_dir + '/' + self.im_list[idx2]):
                    os.makedirs(output_dir + '/' + self.im_list[idx2])
                frame_out = model(self.input0_list[idx], self.input1_list[idx])
                gt = self.gt_list[idx]
                imwrite(frame_out, output_dir + '/' + self.im_list[idx2] + '/' + output_name, range=(0, 1))
                frame_out = func(frame_out)
                gt = func(gt)
                _ = ax[1].imshow(frame_out)
                _ = ax[0].imshow(gt)
                _ = ax[0].set_title("Ground Truth")
                _ = ax[1].set_title("Predicted")
                plt.savefig(os.path.join(output_dir, self.im_list[idx2], f"{self.im_list[idx2]}_{idx}.png"))
                fig.clear()
                psnr = -10 * log10(np.mean((gt - frame_out) * (gt - frame_out)))
                av_psnr += psnr
                msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx2] + ': ', psnr) + '\n'
                print(msg, end='')
                if logfile is not None:
                    logfile.write(msg)
            av_psnr /= len(self.input0_list)
            msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
