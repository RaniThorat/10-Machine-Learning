# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:25:33 2023

@author: Admin
"""

import numpy as np
def gradient_discent(x,y):
    m_curr=b_curr=0
    iterations=10
    n=len(x)
    learning_rate=0.001
    for i in range(iterations):
        y_predicated=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicated)])
        md=-(2/n)*sum(x*(y-y_predicated))
        bd=-(2/n)*sum(y-y_predicated)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("n {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost,i))



x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradient_discent(x, y)
