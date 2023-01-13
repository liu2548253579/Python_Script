
from asyncore import read
from turtle import position
import cv2
import glob
import re
import os
from pyparsing import line_end

dots=0
lines=0
photo_marks=0
stop_100ms_times=0
record_photo_addr_tmp=''
record_photo_addr=''
record_line=''
stoped_lines=''
log_lines=''
previous_record=''
txt_addr = 'C:\\Users\\25482\\Desktop\\connection\\route_txts\\'


txt_path=glob.glob(txt_addr+'*.txt')
with open(txt_path[0],encoding='utf-8') as file:
    content=file.readlines()
for lines in range(len(content)):     
    for line in content[lines]:
        if (line!=',')&(dots==16):
            record_photo_addr_tmp+=line          
        if (line!=',')&(dots==18):
            record_line+=line
        if dots <= 18: 
            previous_record+=line     
        if line == ',':
            dots=dots+1
        if dots == 22: 
            dots=0 
            lines=lines+1 
            if int(record_line)>=40 :stop_100ms_times=stop_100ms_times+1
            if int(record_line)<40 :stop_100ms_times=0   
            if stop_100ms_times==10 :
                stop_100ms_times=0 ;stoped_lines+=str(lines)
                log_lines+=str(photo_marks)+'#lines'+str(lines)+'lines#'+str(photo_marks)+'-'+str(photo_marks)+'#previous'+previous_record+'previous#'+str(photo_marks)
                record_photo_addr+='/*/'+str(photo_marks)+str(lines)+str(photo_marks)+'/*/'+'#*'+str(photo_marks)+record_photo_addr_tmp+str(photo_marks)+'*#'+'\n'
                photo_marks=photo_marks+1
            record_line=''
            record_photo_addr_tmp=''
            previous_record=''
            break




def rewrite_file(re_lines,re_content):
    with open(txt_path[0],'r',encoding='utf-8') as input_file, open('log_file','w',encoding='utf-8') as output_file:
        read_lines=0
        for line in input_file:
            read_lines+=1
            if read_lines==re_lines:
                output_file.write(re_content+'\n')
            else:
                output_file.write(line)
    with open(txt_path[0],'w',encoding='utf-8') as input1_file, open('log_file','r',encoding='utf-8') as output1_file:
        input1_file.write(output1_file.read())
    os.remove('log_file')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


photo = 0 #当前显示的照片标号
marktimes = 0 #鼠标点击次数
key_value = 0 #当前键入的键值
text_position = [0,0,0] #储存四个坐标的数组
text_content = 0 #输入文本的内容
x_position = [0,0,0]#两个X轴坐标数据
y_position = [0,0,0]#两个Y轴坐标数据


def text_write(num,msg):#打开路径下的标号文件并写入内容
    file = open(num,'w')             
    file.writelines(msg) 
    file.close() 
    

def mouse(event, x, y, flags, param):#鼠标点击事件并返回鼠标在图中的坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        global marktimes
        global text_position
        marktimes=marktimes+1
        if marktimes >= 2:
            marktimes = 0
        xy = "%d,%d" % (x, y)
        x_position[marktimes]=x
        y_position[marktimes]=y
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)

        text_position[marktimes]= xy


#fileaddr = input("输入程序文件夹的绝对路径：")           
marktimes=-1
while True :
    img = cv2.imread(record_photo_addr[record_photo_addr.index('#*'+str(photo))+3:record_photo_addr.index(str(photo)+'*#')])
    cv2.namedWindow("image")
    
    image_info="Page:%d click:%d"%(photo,marktimes+1)
    cv2.putText(img, image_info, (5, 30), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), thickness = 1) 
    
    cv2.line(img, (x_position[0],y_position[0]), (x_position[1],y_position[1]), (0, 0, 255))   
    
    x1y1 = "left_down(%d,%d)"%(x_position[0],y_position[0]); x2y2 = "right_up(%d,%d)"%(x_position[1],y_position[1])
    cv2.putText(img, x1y1, (x_position[0],y_position[0]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)
    cv2.putText(img, x2y2, (x_position[1],y_position[1]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)

        
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mouse)
    key_value=cv2.waitKey(10) & 0xFF #读取键入值
    
    
    if key_value == ord('l'):#l键上一张
        photo = photo-1 
        if photo <= 0:
            photo = 0
        x_position[0]=0 ;x_position[1]=0
        y_position[0]=0 ;y_position[1]=0 

        
    if key_value == ord('c'):#c键重新标定当前照片
        x_position[0]=0 ;x_position[1]=0 
        y_position[0]=0 ;y_position[1]=0 
        marktimes=0


    if key_value == ord('n'):#n键下一张
        if marktimes == 1:
            rewrite_line_data=int(log_lines[log_lines.index(str(photo)+'#lines')+len(str(photo)+'#lines'):log_lines.index('lines#'+str(photo))])
            rewrite_previous_data=log_lines[log_lines.index(str(photo)+'#previous')+len(str(photo)+'#previous'):log_lines.index('previous#'+str(photo))]+text_position[0]+','+text_position[1]
            rewrite_file(rewrite_line_data,rewrite_previous_data)
        x_position[0]=0 ;x_position[1]=0 
        y_position[0]=0 ;y_position[1]=0  
        photo = photo+1
        marktimes=-1

    if key_value == 27:#esc键退出
        if marktimes == 1:
            rewrite_line_data=int(log_lines[log_lines.index(str(photo)+'#lines')+len(str(photo)+'#lines'):log_lines.index('lines#'+str(photo))])
            rewrite_previous_data=log_lines[log_lines.index(str(photo)+'#previous')+len(str(photo)+'#previous'):log_lines.index('previous#'+str(photo))]+text_position[0]+','+text_position[1]
            rewrite_file(rewrite_line_data,rewrite_previous_data)
        cv2.destroyAllWindows()
        break