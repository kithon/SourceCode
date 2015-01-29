# -*- coding: utf-8 -*-
from multiprocessing import Process, Array

class Obj(object):
    def __init__(self, num):
        self.num = num

def pro(a):
    print dir(a)

        
def test():
    obj_list = []
    for i in xrange(10):
        o = Obj(i)
        obj_list.append(o)

    array = Array('o', obj_list)
    pro(array)
    
if __name__ == '__main__':
    test()
