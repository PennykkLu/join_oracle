import time
from PIL import Image
from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog

import os
import sys
sys.path.append("..")
from searchFile import find_pic

canvas_width = 1000
canvas_height = 500

class SelectAndSearch(object):
    def __init__(self, master=None):
        self.app = master
        self.app.geometry("1000x500")
        # 启动后创建组件
        self.create()

    def create(self):
        # 创建一个输入框
        path = tk.StringVar()
        img_path = tk.Entry(self.app, font=("宋体", 18), textvariable=path)
        # 顺序布局
        img_path.pack()
        # 坐标
        img_path.place(relx=0.15, rely=0.2, relwidth=0.6, relheight=0.1)


        # 参数是：要适应的窗口宽、高、Image.open后的图片
        # 调整尺寸
        def img_resize(w_box, h_box, pil_image):
            print(pil_image)
            # 获取图像的原始大小
            width, height = pil_image.size
            f1 = 1.0 * w_box / width
            f2 = 1.0 * h_box / height
            factor = min([f1, f2])
            width = int(width * factor)
            height = int(height * factor)
            # 更改图片尺寸，Image.ANTIALIAS：高质量
            return pil_image.resize((width, height), Image.ANTIALIAS)



        #  添加图片图标
        img_t = Image.open('bjtu.jpg')
        img_t_resized = img_resize(0.1 * canvas_width, 0.1 * canvas_height, img_t)
        self.img_t = ImageTk.PhotoImage(img_t_resized)
        lbt = tk.Label(self.app, imag=self.img_t, compound=tk.CENTER,bg='white')
        lbt.place(relx=0.15, rely=0.09, relwidth=0.05, relheight=0.1)

        #  本地上传图标
        upload = tk.Button(self.app, text="select picture", font=("宋体", 15), command=lambda: img_choose(path,self.app))
        upload.place(relx=0.7, rely=0.2, relwidth=0.15, relheight=0.1)


        def img_choose(path, app):
            # 打开文件管理器，选择图片
            selectFileName = filedialog.askopenfilename(parent=app, initialdir='/home/hxxiang/test_pic/', title="本地上传")
            # 同时将图片路径写入行内
            # img_path.delete(0,"end")
            mypath = selectFileName
            path.set(selectFileName)
            # img_path[0] = self.app.picture
            search(mypath, app)


        def search(mypath, app):
            # 被检索的图像路径
            search_path = mypath
            # 存储检索结果
            im_ls=[]
            # 未选择图片，则不检索
            if len(mypath) == 0:
                return
            # 计算检索的耗时
            # 获取当前系统时间
            start = time.perf_counter()

            simi_path = find_pic(search_path)
            simi,im_ls = zip(*simi_path)
            # 获取当前系统时间
            end = time.perf_counter()
            # 计算得到检索所用的总时间
            run_time = end - start
            print('运行时长:', run_time)

            # # 获取相似度
            # score = getScores()

            #  关闭主页面，创建结果界面
            app.destroy()
            result = tk.Tk()
            result.geometry("1000x500")
            result.title('查询结果')
            result.configure(bg='#fffaf0')

            start = time.perf_counter()
            backbutton = tk.Button(result, text="back", font=("宋体", 25), command=lambda: back(result))
            backbutton.place(relx=0.9, rely=0.0, relwidth=0.08, relheight=0.08)

            choosebutton = tk.Button(result, text="select", font=("宋体", 20), command=lambda: img_choose(path,result))
            choosebutton.place(relx=0.02, rely=0.2, relwidth=0.1, relheight=0.08)

            word1 = tk.Label(result, text='searching pic:', font=("宋体", 25), compound=tk.CENTER, bg='#fffaf0')
            word1.place(relx=0.03, rely=0, relwidth=0.5, relheight=0.07)

            word2 = tk.Label(result, text='result:', font=("宋体", 20), compound=tk.CENTER, bg='#fffaf0')
            word2.place(relx=0.15, rely=0.4, relwidth=0.18, relheight=0.07)
            end = time.perf_counter()
            print('布局除图片外时间:', end - start)


            #  上传的图片
            img0 = Image.open(search_path)
            img0_resized = img_resize(0.3 * canvas_width, 0.3 * canvas_height, img0)
            img0 = ImageTk.PhotoImage(img0_resized)
            lb0 = tk.Label(result, imag=img0, compound=tk.CENTER, bg='#fffaf0')
            lb0.place(relx=0.15, rely=0.08, relwidth=0.2, relheight=0.3)

             #十张检索结果图
            imgs = []

            _, search_name = os.path.split(search_path)
            #print(best_name)
            #print(best_name)

            ## 找自己
            my_rank = -1
            for i in range(10):
                repeat_path, best_name = os.path.split(im_ls[i])
                if search_name == best_name:
                    my_rank = i
                    break

            start = time.perf_counter()

            bias = 0
            ddouble = tk.DoubleVar()
            for i in range(2):
                for j in range(5):
                    index = i*5+j
                    if index == my_rank:
                        bias = 1
                    img = Image.open(im_ls[index+bias])
                    print(im_ls[index+bias])
                    img_resized = img_resize(0.19 * canvas_width, 0.2 * canvas_height, img)
                    imgs.append(ImageTk.PhotoImage(img_resized))
                    lb = tk.Label(result, imag=imgs[index], compound=tk.CENTER, bg='#fffaf0')
                    lb.place(relx=j*0.2, rely=0.5+0.25*i, relwidth=0.19, relheight=0.2)
                    ddouble.set(simi[index+bias])
                    lbtext = tk.Label(result, textvariable= ddouble, bg='#fffaf0')
                    lbtext.place(relx=j*0.2, rely=0.7+0.25*i, relwidth=0.18, relheight=0.06)
            end = time.perf_counter()
            print('布检索结果图片外时间:', end - start)
            ##print(search_name)
            ##print(best_name)
            ##如果是裁剪图片则列出排序
            start = time.perf_counter()
            if '_' in search_name:
                str_list = search_name.split(sep='_')
                strp0 = repeat_path + '/' + str_list[0] + '_1.jpg'
                strp1 = repeat_path + '/' + str_list[0] + '_2.jpg'
                strp3 = repeat_path + '/' + str_list[0] + '_4.jpg'
                strp4 = repeat_path + '/' + str_list[0] + '_5.jpg'
                rank = [im_ls.index(strp0), im_ls.index(strp1), im_ls.index(strp3), im_ls.index(strp4)]

                fm1 = tk.Frame(result, bg='#fffaf0')
                fm1.pack(side=tk.TOP, pady=50)
                lbtext0 = tk.Label(fm1, text='pos[0]: rank' + str(rank[0]), bg='#fffaf0', font=("宋体", 20),
                                   compound=tk.CENTER, ).pack()
                lbtext1 = tk.Label(fm1, text='pos[1]: rank' + str(rank[1]), bg='#fffaf0', font=("宋体", 20),
                                    compound=tk.CENTER, ).pack()
                lbtext2 = tk.Label(fm1, text='pos[3]: rank' + str(rank[2]), bg='#fffaf0', font=("宋体", 20),
                                    compound=tk.CENTER, ).pack()
                lbtext3 = tk.Label(fm1, text='pos[4]: rank' + str(rank[3]), bg='#fffaf0', font=("宋体", 20),
                                    compound=tk.CENTER, ).pack()
            end = time.perf_counter()
            print('检索排名布局时间:', end - start)

            result.mainloop()

        #  返回按键
        def back(result):
            # 摧毁当前结果页面
            result.destroy()
            #  创建主界面

            root = tk.Tk(screenName=':12.0')
            root.title('甲骨检索')

            root.configure(bg='#fffaf0')
            title = tk.Label(root, text='Oracle Bone Retrieval System', font=('微软雅黑', 25), bg='#fffaf0',
                             compound=tk.CENTER, )
            title.place(relx=0.2, rely=0.08, relwidth=0.5, relheight=0.15)

            SelectAndSearch(root)
            root.mainloop()