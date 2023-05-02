import argparse
import os
import sys
import math
import glob
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import tkinter as tk
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from pathlib import Path
from os.path import splitext, basename 
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from function import calc_mean_std, normal, coral
from matplotlib import cm
from function import normal
from tqdm import tqdm
from tkinter import ttk, filedialog as fd
from tkinter.messagebox import showinfo, showerror

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
parser.add_argument('--out_vid_dir', type=str,
                    help='Directory path to output the video')

parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

is_GUI = len(sys.argv) == 1
print("GUI MODE: ", is_GUI)


# Advanced options
save_ext='.jpg'
output_path=args.output
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_force_square_transform(original_size, target_size):
    transform_list = []
    if original_size != target_size: 
        transform_list.append(transforms.Resize((target_size, target_size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def test_arbitrary_size_transform(target_width, target_height):
    transform_list = []
    transform_list.append(transforms.Resize((target_width, target_height)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
    
def test_original_transform():
    transform_list = []
    transform_list.append(transforms.Resize((512, 512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
    
def resize_shape(size):
    return size if size % 64 == 0 else ((size // 64) + 1) * 64

test_mode = "arbitrary size" # If we are testing different ways to make the model accept non-square input


def process_img(network, content_img, style_img, content_img_name=None, style_img_name=None, is_video_frame=False):
    
    original_width, original_height = content_img.size
    larger_side = max(content_img.size)
    target_size = resize_shape(larger_side)
    if test_mode == "arbitrary size":
        img_tf = test_arbitrary_size_transform(resize_shape(original_width), resize_shape(original_height))
        content = img_tf(content_img)
    elif test_mode == "force square size":
        img_tf = test_force_square_transform(larger_side, target_size)
        content = img_tf(content_img)
    elif test_mode == "original":
        img_tf = test_original_transform()
        content = img_tf(content_img)
     
    #print(width_ori, height_ori, width_new, height_new)
    style_im = style_img
    style = img_tf(style_im)
    
    #print("Content Shape:", content.shape)
    #print("Style Shape:", style.shape)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    #print(content.shape, style.shape)
        
    with torch.no_grad():
        output = network(content,style)[0]
        output = output.detach().cpu()
    #print(output.shape)
    if is_video_frame:
        # trans = transforms.ToPILImage()
        grid = make_grid(output[0], normalize = True)
        img = transforms.ToPILImage()(grid).resize((original_width, original_height))
        arr = np.array(img)
        return arr
    else:
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, content_img_name,
            style_img_name, save_ext
        )
        grid = make_grid(output[0], normalize = True)
        img = transforms.ToPILImage()(grid).resize((original_width, original_height))
        
        arr = np.array(img)
        #res_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) 
        res_img = Image.fromarray(arr.astype('uint8'), 'RGB')
        res_img.save(output_name)
        if is_GUI:
            display_img(img, output_file_box)
        return None

def main():
    # Either --content or --content_dir should be given.
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*.jpg')] + [f for f in content_dir.glob('*.png')]  +[f for f in content_dir.glob('*.mp4')]

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
    print("Test Mode: ", test_mode)
    count_vids = 0
    for content_path in content_paths:
        content_name = splitext(basename(content_path))[0]
        content_ext = splitext(content_path)[-1]
        for style_path in style_paths:
            style_img = Image.open(style_path).convert("RGB")
            style_name = splitext(basename(style_path))[0]
            print(content_path, style_path)
            if content_ext == ".mp4":
                capture = cv2.VideoCapture(str(content_path))
                fps = capture.get(cv2.CAP_PROP_FPS)
                print("FPS:", fps)
                count_vids += 1
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                #width, height = resize_shape(width, height)
                print("Video Shape:", width, height)
                output_video_name = str(output_path) + "/result_" + str(count_vids) + ".mp4"
                video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                
                for i in tqdm(range(length)):
                    success, frame = capture.read()
                    if success:
                        # frame: uint8
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        content_img = Image.fromarray(frame)
                        #processing
                        res_img = process_img(network, content_img, style_img, content_img_name=None, style_img_name=style_name, is_video_frame=True)

                        # uint8
                        # clip or rgb here
                        video.write(res_img)
                    else:
                        break
                cv2.destroyAllWindows()
                video.release()
                capture.release()
                result_thumbnail = extract_video_thumbnail(output_video_name)
                display_img(result_thumbnail, output_file_box)
            else:
                content_img = Image.open(content_path).convert("RGB")
                process_img(network, content_img, style_img, content_img_name=content_name, style_img_name=style_name, is_video_frame=False)
    print('----Conversion Finished----')
    showinfo("Info", message="Conversion Successful!")

def select_files(is_style, img_box):
    if is_style:
        filetypes = (
            ('image file', '*.jpg'),
            ('image file', '*.png')
        )
    else:
        filetypes = (
            ('image file', '*.jpg'),
            ('image file', '*.png'),
            ('video file', '*.mp4')
        )
    
    if is_style:
        file_result = fd.askopenfilenames(
            title='Open Input file',
            initialdir='/',
            filetypes=filetypes)
        if len(file_result) != 0:
            args.style = file_result[0]
            img = Image.open(str(args.style))
            display_img(img, img_box)
    else:
        file_result = fd.askopenfilenames(
            title='Open Input file',
            initialdir='/',
            filetypes=filetypes)
        if len(file_result) != 0:
            args.content = file_result[0]
            if splitext(args.content)[-1] == ".mp4":
                img = extract_video_thumbnail(str(args.content))
            else:
                img = Image.open(str(args.content))
            display_img(img, img_box)
  
def extract_video_thumbnail(path):
    capture = cv2.VideoCapture(path)
    _, frame = capture.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    capture.release()
    return img
  
def display_img(image, img_box):
    display_height = 125
    scale = display_height / image.height
    image = ImageTk.PhotoImage(image.resize((math.floor(image.width * scale), display_height)))
    img_box.config(image=image)
    img_box.image = image

def report_callback_exception(self, exc, val, tb):
    print(str(val))
    showerror("Error", message=str(val))
    quit()

root = tk.Tk()
root.title('CS 543 StyTR-2')
root.resizable(False, False)
root.geometry('450x510')
#tk.Tk.report_callback_exception = report_callback_exception
                   
input_button = ttk.Button(
    root,
    text='Input File',
    command=lambda: select_files(False, input_file_box)
)

style_button = ttk.Button(
    root,
    text='Style Image',
    command=lambda: select_files(True, style_img_box)
)

convert_button = ttk.Button(
    root,
    text='Start',
    command=main
)

quit_button = ttk.Button(
    root,
    text="Quit",
    command=quit
)

input_file_box = tk.Label(root)
style_img_box = tk.Label(root)
output_file_box = tk.Label(root)

input_file_box.place(x=190,y=5)
style_img_box.place(x=190,y=140)
output_file_box.place(x=190,y=275)
#input_button.pack(expand=True)
input_button.grid(column=0, row=0, padx=30, pady=(100, 40))
style_button.grid(column=0, row=1, padx=30, pady=40)
convert_button.grid(column=0, row=2, padx=30, pady=40)
quit_button.grid(column=0, row=3, padx=30, pady=40)

if __name__ == "__main__":
    if is_GUI:
        root.mainloop()
    else:
        main()