import argparse
from pathlib import Path
import os
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
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
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
    content_paths = [f for f in content_dir.glob('*')]

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

def process_img(content_path, style_path, frame=None):
    width_c, height_c = Image.open(content_path).size
    print(width_c, height_c)
    size_max = max(width_c, height_c)
    if size_max % 64 != 0:
        start = 64
        while start < size_max:
            start += 64
        size_max = start
    content_im = expand2square(Image.open(content_path).convert("RGB"), (0, 0, 0)).resize((size_max, size_max), Image.LANCZOS)
    style_im = Image.open(style_path).convert("RGB")
    content_tf = test_transform(size_max, size_max)
    style_tf = test_transform(size_max, size_max)
    content_tf1 = content_transform()   
    if frame:
        content = content_tf(frame)
    else:
        content = content_tf(content_im)

    h,w,c=np.shape(content)    
    print(h, w, c)
    style_tf1 = style_transform(h,w)
    style = style_tf(style_im)
    print(style.shape)
  
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    
        
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
        print(output.shape, "!!!!")
        print(width_c, height_c)
        if width_c > height_c:
            start = (width_c - height_c) // 2
            #print(start, start + width_c)
            output = output[:, :, start : (start + height_c), :]
        else:
            start = (height_c - width_c) // 2
            output = output[:, :, :, start : (start + width_c)]
        print(output.shape, "!!!!")
        save_image(output, output_name)
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
            width, height = Image.open(style_path).size
            print("Video Shape:", height, width)
            video = cv2.VideoWriter(str(output_path) + "/result_" + str(count_vids) + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (content_size,content_size))
            
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

   

