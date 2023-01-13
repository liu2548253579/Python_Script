from inspect import trace
from PIL import Image
import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join

txt_num=1
classes = ['light']

def voc_xml_extract(txt_fpath, jpg_fpath, classes):



# 读取txt文件
        with open(txt_fpath, 'r') as f:
            lines = f.readlines()
            lines1 = [line.replace('\n','').split() for line in lines]
            lines2 = []
        for line in lines1:
            classname = classes[int(line[0])]
            print(line[1:])
            xywh = [int(float(x)) for x in line[1:]]
            tem_res = [classname, xywh]
            lines2.append(tem_res)

# img.crop
# left：与左边界的距离
# up：与上边界的距离
# right：还是与左边界的距离
# below：还是与上边界的距离
# 简而言之就是，左上右下。
        with open(jpg_fpath, 'rb') as f:
            img = Image.open(f)
            for i in range(len(lines2)):
                img_crop = img.crop(lines2[i][1])
                img_crop.save("C://Users//25482//Desktop//训练//trafficlight_国内-400//light_img//koutu//%s.jpg"%(txt_num))



while True:
    try:
        voc_xml_extract("C://Users//25482//Desktop//训练//trafficlight_国内-400//light_label//txt//%s.txt"%(txt_num),"C://Users//25482//Desktop//训练//trafficlight_国内-400//light_img//%s.jpg"%(txt_num),classes=classes)

    except : 
        None
            
    txt_num=txt_num+1

    if(txt_num==401): 
        break