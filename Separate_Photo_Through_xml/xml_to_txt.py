from inspect import trace
from PIL import Image
import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join

file_num=1
classes = ['light']

# 读取xml文件

def voc_xml_extract(xml_fpath, txt_fpath, classes):


            # 一次读入xml的ElementTree
        with open(xml_fpath) as f:
            tree = ET.parse(f)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

                # 循环的将标记目标存入输出文件
        with open(txt_fpath, 'w') as f:
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                clsname = obj.find('name').text
                if clsname not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(clsname)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
                bb = (b[0] ,b[2],b[1],b[3] )
                f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        return True 


while True:
    try:
        voc_xml_extract("%s.xml"%(file_num),"txt//%s.txt"%(file_num),classes=classes)

    except : 
        None
            
    file_num=file_num+1

    if(file_num==401): 
        break
