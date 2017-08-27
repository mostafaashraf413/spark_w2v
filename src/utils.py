# -*- coding: utf-8 -*-
#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')

import re

not_arChars = r'[^ـ^\u0600-\u06FF^\u0750-\u077F^\u08A0-\u08FF^\uFB50-\uFDFF^\uFE70-\uFEFF]'
def extract_arabic(line):
    global not_arChars
    
    line = re.sub(not_arChars,u' ',line)
    line = re.sub(r'\s+',u' ',line)
    line = line.strip()
    #print(line)
    return (line.split(" "),)