from voidMap import VoidMap
from simplexTools import *

voidMap = VoidMap("voido",3,
                  Simplex(noise_f, x_offset=300, y_offset=7000, frequency=0.002),
                  20,170,0.5,
                  Simplex(noise_f, x_offset=999, y_offset=1000, frequency=0.008),
                  Simplex(noise_f, x_offset=999, y_offset=7200, frequency=0.008),
                  112385,6158639,
                  [(-768,-768),(-768,1280),(1280,-768),(1280,1280)],
                  150,500)

assert False

voidMap = VoidMap("voido",1,
                  Simplex(noise_f, x_offset=300, y_offset=7000, frequency=0.002),
                  20,170,0.5,
                  Simplex(noise_f, x_offset=999, y_offset=1000, frequency=0.008),
                  Simplex(noise_f, x_offset=999, y_offset=7200, frequency=0.008),
                  112385,6158639,
                  [(256,256)],
                  150,500)

assert False



