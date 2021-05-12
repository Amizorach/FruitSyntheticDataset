from PIL import Image, ImageDraw
import random
from PIL import ImageFilter
from PIL import ImageColor
from pascal import PascalVOC, PascalObject, BndBox, size_block
from pathlib import Path
import cv2
import numpy as np
import os
import getopt, sys
import argparse

# bg_colors = []
# fg_colors = []

img_path = 'train/images'
ann_path = 'train/ann'
width = 250
height = 250

# pobjs = []
def prepare_colors():
    txt_leaves = ['#608d2a', '#a8b146', '#ccf0bc', '#aace57', '#2e5d0b', '#071d06', '#71a476', '#3a4533', '#759a65', '#25321f', '#59734d', '#90ab64', '#7b8554', '#7b8554', '#4d5430']
    txt_sky = ['#e9e3c3', '#99949e', '#9bb5cf' , '#f6fbfb', '#91959b', '#c0e1f9', '#dd9b98', '#dd9b98', '#deb29f']
    txt_ground = ['#3d2c15', '#dfcba6', '#8d6e42', '#c99276', '#655223', '#dfd1c1', '#f5c9ae', '#9a8149']

    bg_colors = []
    fg_colors = []

    for t in txt_leaves:
        bg_colors.append(ImageColor.getrgb(t))
        fg_colors.append(ImageColor.getrgb(t))
    for t in txt_sky:
        bg_colors.append(ImageColor.getrgb(t))
    for t in txt_ground:
        bg_colors.append(ImageColor.getrgb(t))

    return bg_colors, fg_colors

# def create_images():
#     im_bg = Image.new('RGBA', (width, height), ImageColor.getrgb('#7FCBFDFF'))
#     im_fg = Image.new('RGBA', (width, height), (0, 0, 0, 0))
#     im_oranges = Image.new('RGBA', (width, height), (0, 0, 0, 0))
#     return im_bg, im_fg, im_oranges

def plot_random_blobs(draw, colors, count, mins, maxs):
    for i in range(count):
        x = random.randint(0,width)
        y = random.randint(0,height)
        w = random.randint(mins,maxs)
        l = random.randint(mins,maxs)
        c = colors[random.randint(0,len(colors)-1)]
        draw.ellipse((x, y, x+w, y+l), fill=c, outline=None)

def create_bg(colors, width, height):
    im_bg = Image.new('RGBA', (width, height), ImageColor.getrgb('#7FCBFDFF'))
    draw_bg = ImageDraw.Draw(im_bg)
    plot_random_blobs(draw_bg, colors, 1500, 10, 25)
    im_bg = im_bg.filter(ImageFilter.MedianFilter(size=9))
    return im_bg

def create_fg(colors, width, height):
    im_fg = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_fg = ImageDraw.Draw(im_fg)
    plot_random_blobs(draw_fg, colors, 40, 10, 25)
    im_fg = im_fg.filter(ImageFilter.MedianFilter(size=9))
    return im_fg

def plot_random_fruit(color_range, count, width, height, mins, maxs):
    im_fruit = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_fruit = ImageDraw.Draw(im_fruit)

    fruit_info = []
    for i in range(count):
        x = random.randint(0,width-10)
        y = random.randint(0,height-10)
        w = random.randint(mins,maxs)
        c = (random.randint(color_range[0][0],color_range[0][1]),
             random.randint(color_range[1][0], color_range[1][1]),
             random.randint(color_range[2][0], color_range[2][1]))
        fruit_info.append([x, y, w, w, c])
        draw_fruit.ellipse((x, y, x+w, y+w), fill=c, outline=None)
    return im_fruit, fruit_info

def create_layered_image(im_bg, im_fruit, im_fg):
    img = im_bg.copy()
    img.paste(im_fruit, (0, 0), im_fruit)
    img.paste(im_fg, (0, 0), im_fg)
    return img

def create_annotation(img, fruit_info, obj_name,
                      img_name ,ann_name):
    pobjs = []
    for i in range(len(fruit_info)):
        pct = 0
        circle = fruit_info[i]
        color = circle[4]
        for i in range(circle[2]):
            if (circle[0]+i >= width):
                continue;
            for j in range(circle[3]):
                if (circle[1]+j >= height):
                    continue;
                r = img.getpixel((circle[0]+i, circle[1]+j))
                if (r[0] == color[0]):
                    pct = pct +1
        diffculty = pct/(circle[2]*circle[3])

        if (diffculty > 0.1):
            dif = True
            if (diffculty > 0.4):
                dif = False
            pobjs.append(
                PascalObject(obj_name, "", truncated=False,
                             difficult=dif,
                             bndbox=BndBox(circle[0], circle[1],
                                           circle[0]+circle[2],
                                           circle[1]+circle[3])))
    pascal_ann = PascalVOC(img_name,
                           size=size_block(width, height, 3),
                           objects=pobjs)
    pascal_ann.save(ann_name)

def plot_image(img_name, ann_name):
    ann = PascalVOC.from_xml(ann_name)
    img = cv2.imread(img_name)
    for obj in ann.objects:
        p1 = (obj.bndbox.xmin, obj.bndbox.ymin)
        p2 = (obj.bndbox.xmax, obj.bndbox.ymin)
        p3 = (obj.bndbox.xmax, obj.bndbox.ymax)
        p4 = (obj.bndbox.xmin, obj.bndbox.ymax)
        cv2.line(img, p1, p2, color=(0, 255, 0), thickness=1)
        cv2.line(img, p2, p3, color=(0, 255, 0), thickness=1)
        cv2.line(img, p3, p4, color=(0, 255, 0), thickness=1)
        cv2.line(img, p4, p1, color=(0, 255, 0), thickness=1)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def create_training_image(counter, bg_colors, fg_colors,
                          fruit_color_range):
    fruit_count = random.randint(0, 20)
    ext = '{}_{}'.format(counter, fruit_count)
    img_name = '{}/fruit_{}.png'.format(img_path, ext)
    ann_name = '{}/ann_{}.xml'.format(ann_path, ext)

    im_bg = create_bg(bg_colors, width, height)
    im_fg = create_fg(fg_colors, width, height)
    im_fruit, fruit_info = plot_random_fruit(fruit_color_range,
                                             fruit_count, width, height, 10, 25)
    img = create_layered_image(im_bg, im_fruit, im_fg)

    #create the anootation File
    create_annotation(img, fruit_info, 'oranges',
                      img_name, ann_name)
    img.save(img_name)
    return img, img_name, ann_name

def create_training_set(num, start_at=0, plot=False):
    bg_colors, fg_colors = prepare_colors()
    fruit_color_range = [[180,230],[50,130],[0,5]]
    for i in range(num):
        _, img_name, ann_name = create_training_image(i+start_at, bg_colors,
                              fg_colors, fruit_color_range)
        if (plot):
            plot_image(img_name, ann_name)
        print('{}:{}'.format(i, img_name))




def main():
    global img_path
    global ann_path
    global width
    global height

    startat = 0
    plot = False
    short_options = "s:pc:d:a:"
    long_options = ["count=", "plot", "size=", "path=", "startat="]
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=250, required=False)
    parser.add_argument('-p', '--plot', type=bool, default=False, required=False)
    parser.add_argument('-c', '--count', type=int, default=1, required=False)
    parser.add_argument('-d', '--dir', type=str, default='train', required=False)
    parser.add_argument('-a', '--startat', type=int, default=0, required=False)

    args = parser.parse_args()
    width = args.size
    height = args.size
    tot_count = args.count
    startat = args.startat
    plot = args.plot
    p = args.dir
    img_path = '{}/images'.format(p)
    ann_path = '{}/ann'.format(p)
    try:
        os.makedirs(img_path,  exist_ok=True)
        os.makedirs(ann_path,  exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Cant create directories {} {}".format(img_path, ann_path))
            raise
    # full_cmd_arguments = sys.argv
    # argument_list = full_cmd_arguments[1:]
    # try:
    #     arguments, values = getopt.getopt(argument_list, short_options, long_options)
    # except getopt.error as err:
    #     # Output error, and return with an error code
    #     print (str(err))
    #     sys.exit(2)
    # for current_argument, current_value in arguments:
    #     if current_argument in ("-s", "--size"):
    #         width = current_value
    #         height = current_value
    #         print ('size set to ({},{})'.format(width, height) )
    #     elif current_argument in ("-p", "--plot"):
    #         plot = True
    #         print ("plot set to True")
    #     elif current_argument in ("-c", "--count"):
    #         tot_count = current_value
    #         print ("count set to {}".format(current_value))
    #     elif current_argument in ("-d", "--path"):
    #         path = current_value
    #         print ("path set to {}".format(current_value))
    #         img_path = '{}/images'.format(path)
    #         ann_path = '{}/images'.format(path)
    #     elif current_argument in ("-a", "--startat"):
    #         print ("startat set to {}".format(current_value))
    #         startat = current_value
    create_training_set(tot_count, startat, plot)


if __name__ == "__main__":
    main()
# Output argument-wise
