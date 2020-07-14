import re
import os
import pandas as pd
import xml.dom.minidom as dm
import pickle

'''
path = 'news_tensite_xml.dat'

def split():
    p = re.compile('</doc>',re.S)
    end = '</doc>'

    fileContent = open(path,'r',encoding='GB18030-2000').read()

    paraList = p.split(fileContent)

    fileWriter = open('./files/0.txt', 'a', encoding='utf-8')

    #遍历切片后的文本列表
    for paraIndex in range(len(paraList)):
        fileWriter.write(paraList[paraIndex])
        if(paraIndex != len(paraList)):
            fileWriter.write(end)

        if((paraIndex + 1) % 5000 == 0):
            fileWriter.close()
            fileWriter = open('./files/' + str((paraIndex + 1) / 5000) + '.txt', 'a', encoding='utf-8')

    fileWriter.close()
    print('finished')
'''

def file_fill(file_dir):

    count = 0
    for filename in os.listdir('./files/'):
        count += 1
        #print(filename)
    print(count)

    for i in range(count - 1):

        path = './files/%.1f.txt' % (i)

        file = open(path, 'r', encoding='utf-8')

        ok_path = './news_after2/%d.txt' % (i)
        ok_file = open(ok_path, 'a+', encoding='utf-8')

        get_list = re.findall(r"<contenttitle>(.*?)</contenttitle>\n<content>(.*?)</content>",file.read())

        result_list = []

        for result_tuple in get_list:
            if '第一次登陆将自动注册为本网用户' not in result_tuple[0]:
                result_list.append(result_tuple)

        for result in result_list:
            ok_file.write('标题:' + result[0])
            ok_file.write('\n')
            ok_file.write('正文:' + result[1])
            ok_file.write('\n\n')

        file.close()
        ok_file.close()

file_fill('files')



