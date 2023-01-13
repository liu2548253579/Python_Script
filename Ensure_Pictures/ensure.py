'''
Author: Frank Liu ljq.frank@qq.com
Date: 2023-01-13 12:24:31
LastEditors: Frank Liu ljq.frank@qq.com
LastEditTime: 2023-01-13 12:25:25
FilePath: \undefinedd:\LEARNINGRESOURCES\Git\Python_Script\mark_pictures\ensure.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import glob
import re

photo = 0 #当前显示的照片标号
key_value = 0 #当前键入的键值
text_content = " " #输入文本的内容
#fileaddr = 'C:\\Users\\25482\\Desktop\\'
#photoaddr = 'C:\\Users\\25482\\Desktop\\ensure_pictures\\pictures\\'

def text_create(name):#创建指定文件的文本(如若存在则会覆盖原文件)
    text_path = fileaddr+'\\ensure_pictures\\texts\\'    
    full_path = text_path + name + '.txt' 
    file = open(full_path,'w')             
    file.close() 


def text_write(num,msg):#打开路径下的标号文件并写入内容
    file = open(fileaddr+'\\ensure_pictures\\texts\\'+txt_name+'.txt','w')             
    file.write(msg) 
    file.close() 

fileaddr = input("输入程序文件夹的绝对路径：")           
photoaddr = input("输入图片的绝对路径：")

image_photo=glob.glob(photoaddr+'\\*.png')

while True :
    global txt_name
    txt_name=str(re.findall(r'L_([^"]+).png',image_photo[photo]))
    
    img = cv2.imread(image_photo[photo])
    cv2.namedWindow("image")
    
    image_info="Page:%d state:%s"%(photo,text_content)
    cv2.putText(img, image_info, (5, 30), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), thickness = 1) 
    
    cv2.imshow("image", img)
    key_value=cv2.waitKey(10) & 0xFF #读取键入值
    
    
    if key_value == ord('l'):#l键上一张
        photo = photo-1 
        if photo <= 0:
            photo = 0
        text_create(txt_name) # 创建文本(此处是为了清除错误输入) 
        
    if key_value == ord('c'):#c键重新标定当前照片
        text_content=" "       
            
    if key_value == ord('n'):#n键下一张
        text_create(txt_name) # 创建文本
        text_write(txt_name,text_content)#写入内容
        photo = photo+1
        text_content = " "


    if key_value == ord('t'):#t键表示正确
        text_content="true"

    if key_value == ord('f'):#f键表示错误
        text_content="false"
        
    if key_value == 27:#esc键退出
        text_create(txt_name) # 创建文本
        text_write(txt_name,text_content)#写入内容
        cv2.destroyAllWindows()
        break