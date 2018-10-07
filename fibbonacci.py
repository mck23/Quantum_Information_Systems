#!/usr/bin/env python

a=1
b=1

bound = input("Find Fibbonacci Numbers less than this (please enter a postive integer) ")
try:
    bound  = int(bound)
except ValueError:
    print("That's not an int! So you won't find all you want, so you only get 1")
    bound=1
if (bound > 0) :
    print(a," ", end="")
    while (b < bound) :
        print(b," ", end="")
# Normally you might think assigment might be something a below, but python has 
        a,b = b, a+b
#       temp = b
#       b = b+a
#       a = temp	
else:
   print("Now don't be difficult, be positive")

print()

