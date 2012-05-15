#!/usr/bin/python

from math import *
    

# Based on:
# http://en.wikipedia.org/wiki/Golden_section_search

phi = (1 + sqrt(5)) / 2
resphi = 2 - phi

# x1 and x3 are the current bounds; the minimum is between them.
# x2 is the center point, which is closer to x1 than to x3
def goldenSectionSearch(f, x1, x2, x3, fx2, tau, max_steps):
 
    # Create a new possible center in the area between x2 and x3, closer to x2
    x4 = x2 + resphi * (x3 - x2)
 
    # Evaluate termination criterion
    if abs(x3 - x1) < tau * (abs(x2) + abs(x4))/2 or max_steps==0:
        return (x3 + x1) / 2
  
    fx4=f(x4)
    if fx4 >= fx2:
        return goldenSectionSearch(f, x2, x4, x3, fx4, tau, max_steps-1)
    else:
        return goldenSectionSearch(f, x4, x2, x1, fx2, tau, max_steps-1)

def findMax(f, min, max, precision=1e-3,max_steps=-1):
    """Find maximum of function f, 
       f .. function to maximize
       min .. start of the search interval
       max .. end of the search interval
       precision .. desired precision of x
       max_steps .. maximum number of search steps"""
    x2 = min + resphi * (max-min)
    return goldenSectionSearch(f, min, x2, max, f(x2), sqrt(precision), max_steps)
    


if __name__ == "__main__":
    N=0
    def f(x):
        global N
        N += 1

        y = 10 -x*x -2*x
        print N,'(',x,',',y,')'
        return y

    N=0
    print "Driven by precision"
    print findMax(f,-20,20)
    
    N=0
    print "Driven by max steps=16"
    print findMax(f,-20,20,max_steps=16)


