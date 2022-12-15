import sys
sys.path.append('../')
sys.path.append('./')
import os
import json
import pprint
if __name__ == '__main__':
    dir = "./JSON/LIW_2007" #Directory of attack results stored by ML-BA
    jsonlist = os.listdir(dir)
    succ = 0
    l1_norm=0
    l2_norm=0
    infiy_norm=0
    rmsd=0
    mean=0
    count=0
    for json_name in jsonlist:
        read = json.load(open(dir+"/"+json_name, 'r', encoding="utf-8"))
        count+=1
        if read['l2_norm']>0 and read['l2_norm']<=100:
            succ+=1;
            l1_norm+=read['l1_norm']
            l2_norm += read['l2_norm']
            infiy_norm += read['infiy_norm']
            rmsd += read['rmsd']
            mean += read['mean']

    print("count = ", count)
    print("succ = ", succ)
    print("succ_rate = " ,succ/count)
    print("l1_norm = ", l1_norm / succ)
    print("l2_norm = ", l2_norm / succ)
    print("infiy_norm = ", infiy_norm / succ)
    print("rmsd = ", rmsd / succ)
    print("mean = ", mean / succ)


