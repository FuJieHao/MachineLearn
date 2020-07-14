
#读文件
txt = open('./fj.txt')
txt_read = txt.read()

print(txt_read)

txt.close()
txt = open('./fj.txt')

lines = txt.readlines()
print(lines)

for line in lines:
    print('cur_line:',line)

txt.close()


#写文件  w 覆盖   a 追加
txt = open('fj_write.txt','w')
txt.write('jin tian tian qi bu cuo\n')
txt.write('hao fu jie')
txt.close()


txt = open('fj_write.txt','w')
try:
    for i in range(10):
        10/(i-5)
        txt.write(str(i) + '\n')
except Exception:
    print('error:',i)
finally:
    txt.close()

# 自动执行except  和 finally
with open('fj_write.txt','w') as f:
    f.write('jin tian tian qi bu cuo')