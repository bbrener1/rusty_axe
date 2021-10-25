from rusty_axe import lumberjack
import numpy as np

# Trivial test:

test = np.arange(9).reshape((3,3))

lumberjack.fit(test)
