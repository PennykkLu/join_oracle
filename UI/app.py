from SelectAndSearch import *

import sys
sys.path.append("..")
from searchFile import find_pic

#path = '/home/hxxiang/test_pic/100_3.jpg'
#find_pic(path)



root = tk.Tk(screenName = ':12.0')
root.title('甲骨检索')

root.configure(bg='#fffaf0')
title = tk.Label(root, text='Oracle Bone Retrieval System', font=('微软雅黑',25), bg='#fffaf0',compound=tk.CENTER,)
title.place(relx=0.2, rely=0.08, relwidth=0.5, relheight=0.15)

SelectAndSearch(root)
root.mainloop()