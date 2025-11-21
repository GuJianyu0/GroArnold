#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tables, struct
import io, sys
reload(sys)
sys.setdefaultencoding('utf-8')

#write binary
def write_b():
    f = open("20201111_readbin.b","wb")
    f.write(b'\xd6\xd0\xb9\xfa')
    s = '\n你好\n'
    length = f.write(s.encode('utf-8'))
    print("write: {}-- write in: {} length".format(s,length))
    f.close()

if __name__ == "__main__":
    fname = "gadget/dice/example/galaxy_PM17.g1"
    f = open(fname, 'rb')
    # print f.read()
    # for i in range(96):
    #     for j in range(10000):
    #         data = f.read(8)
    #         elem = struct.unpack("f", data)[0]
    f.close()

    write_b()