#假设学校总人数是U
#穿长裤的男生:   U * (Boy) * P(Pants | Boy)
#穿长裤的女生:   U * (Girl) * P(Pants | Girl)

#穿长裤的总数:U * (Boy) * P(Pants | Boy) + U * (Girl) * P(Pants | Girl) = T

#现在求 P(Girl | Pants) = U * (Girl) * P(Pants | Girl) / T
#这样可以约掉总人数U
#分母就是P(Pants)
#分子就是P(Pants | Girl)

#贝叶斯公式
#P(A | B) = [P(B | A) * P(A)] / P(B)
