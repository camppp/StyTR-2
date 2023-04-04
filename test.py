import argparse
from pathlib import Path
import os
import math
import glob
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
import cv2

def test_transform(width_new, height_new):
    transform_list = []
    if width_new and height_new: 
        transform_list.append(transforms.Resize((height_new, width_new)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')
parser.add_argument('--out_vid', type=str,
                    help='Flag representing if going to generate video')
parser.add_argument('--out_vid_dir', type=str,
                    help='Directory path to output the video')

parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()




# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    if args.out_vid:
        content_paths = [f for f in content_dir.glob('*.mp4')]
    else:
        content_paths = [f for f in content_dir.glob('*.jpg')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()

state_dict = torch.load(args.decoder_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
network.eval()

network.to(device)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize_shape(width, height):
    res_width = width
    res_height = height
    if width % 64 != 0:
        start = 64
        while start < width:
            start += 64
        res_width = start
    if height % 64 != 0:
        start = 64
        while start < height:
            start += 64
        res_height = start
    return res_width, res_height
        
def process_img(content_path, style_path, frame=None):
    if frame:
        width_ori, height_ori = frame.size
        width_new, height_new = resize_shape(width_ori, height_ori)
        content_tf = test_transform(width_new, height_new)
        content = content_tf(frame.convert("RGB"))
    else:
        img = Image.open(content_path)
        width_ori, height_ori = img.size
        width_new, height_new = resize_shape(width_ori, height_ori)
        content_tf = test_transform(width_new, height_new)
        content = content_tf(img.convert("RGB"))
    #print(width_ori, height_ori, width_new, height_new)
    style_im = Image.open(style_path).convert("RGB")
    
    style_tf = test_transform(width_new, height_new)
    h,w,c=np.shape(content)    
    
    style = style_tf(style_im)
    #print("Content Shape:", content.shape)
    #print("Style Shape:", style.shape)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    #print(content.shape, style.shape)
        
    with torch.no_grad():
        output = network(content,style)[0]
        output = output.cpu()
    #print(output.shape)
    if frame:
        # trans = transforms.ToPILImage()
        grid = make_grid(output[0],normalize = True)
        return grid
        # return trans(output[0])
    else:
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )
        transform = transforms.ToPILImage()
        img = transform(output[0])
        img = img.resize((width_ori, height_ori))
        img.save(output_name)
        return None

count_vids = 0
for content_path in content_paths:
    for style_path in style_paths:
        print(content_path, style_path)
        if args.out_vid == "true":
            capture = cv2.VideoCapture(str(content_path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            print("FPS:", fps)
            frameNr = 0
            count_vids += 1
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #width, height = resize_shape(width, height)
            print("Video Shape:", width, height)
            video = cv2.VideoWriter(str(output_path) + "/result_" + str(count_vids) + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            while (True):
                print(frameNr)
                success, frame = capture.read()
                if success:
                    # frame: uint8
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    im_pil = Image.fromarray(frame)
                    #processing
                    temp = process_img(content_path, style_path, im_pil)
                    #1,0;3,512,512
                    temp = transforms.ToPILImage()(temp)
                    temp = np.array(temp)
                    im_write = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR) 
                    im_write = cv2.resize(im_write, (width, height))
                    # 

                    # 512,512,3
                    # uint8
                    # clip or rgb here
                    video.write(im_write)
                else:
                    break
                frameNr = frameNr+1
            
            cv2.destroyAllWindows()
            video.release()
            capture.release()
            
        else:
            process_img(content_path, style_path)
print('end')

   

