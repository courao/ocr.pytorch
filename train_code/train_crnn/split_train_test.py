import numpy as np
h1 = open('infofiles/infofile_selfcollect.txt',encoding='utf-8')
h2 = open('infofiles/infofile_selfcollect_train.txt','w',encoding='utf-8')
h3 = open('infofiles/infofile_selfcollect_test.txt','w',encoding='utf-8')
content = h1.readlines()
for line in content:
    if np.random.random()<0.05:
        h3.write(line)
    else:
        h2.write(line)
