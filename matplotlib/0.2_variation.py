import matplotlib.pyplot as plt

# r--> red and o is the shape
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.grid()
plt.savefig("0.2_ro_plot.png")

import numpy as np
t = np.arange(0., 10., 0.5) 

plt.plot(t, t**2, 'b--', label='^2') # blue dashes, label for showing up in legend
plt.plot(t,t**2.2, 'rs', label='^2.2') # red squares
plt.plot(t, t**2.5, 'g^', label='^2.5') # green triangles
# multiple plots for multiple lines
plt.grid()
plt.legend() # add legend
plt.savefig("0.2_blueDashes_redSquares_greenTriangles_plot.png")
