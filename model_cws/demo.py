#使用工具运行
from model_cws.medical_cws import medical_seg

my_pred=medical_seg()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
result=my_pred.predict_sentence("".join(sentence.split()))
print('处理完：'+result)
#
# 输入文件(测试文件，输出文件)
# my_pred.predict_file("D:\\CMe_KG\\CMeKG\\model_cws\\test\\my_test.txt","D:\\CMe_KG\\CMeKG\\model_cws\\test\\outt.txt")