'''
Miscellaneous tools / utilities / helper methods.
'''

from __init__ import *


def any_value(my_dict):
    for (k, v) in my_dict.items():
        if v is not None:
            return v
    return None
