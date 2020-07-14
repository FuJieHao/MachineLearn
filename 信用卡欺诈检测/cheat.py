#for root, dirs, files in os.walk(file_dir):  #扫描该目录下的文件夹和文件,返回根目录路径,文件夹列表,文件列表
    
    
    
    #'''
    #print(files)
    '''
        for f in files:
        if f == '.DS_Store':
        continue
        tmp_dir = './news_after2' +'/' + f #加上标签后的文本
        
        text_init_dir = file_dir + '/' + f #原始文本
        #print(tmp_dir)
        file_source = open(text_init_dir, 'r', encoding='utf-8') #打开文件,并将字符按照utf-8编码,返回unicode字节流
        #print(file_source)
        ok_file = open(tmp_dir, 'a+', encoding='utf-8')
        start_title = '<contenttitle>'
        end_title = '</contenttitle>'
        start_content = '<content>'
        end_content = '</content>'
        line_content = file_source.readline() #按行进行读取
        get_list = re.findall(r"<contenttitle>(.*?)</contenttitle>\n<content>(.*?)</content>",file_source.read())
        
        print(get_list)
        
        for result in get_list:
        print(result[0])
        break
        '''
            '''
                #for result in get_title_list, get_content_list:
                #    print(result)
                '''
            for get_title in get_title_list:
                result_title = re.match(start_title + '.*?' + end_title, get_title)
                print(result_title)
            '''
                #print(file_source.read()[start_title:end_title])
                '''
            ok_file.write(start_title)
            for lines in line_content:
                text_temp = lines.replace('&', '.')
                text = text_temp.replace('', '')
                ok_file.write(text)
        
            ok_file.write('\n' + end_title)
            
            file_source.close()
            ok_file.close()
            '''
                
                #print('finished')
