import matplotlib.pyplot as plt
import numpy as np

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.
plt.subplot(211) # 2 --> rows. 1 -- column. 1 --> 1st subplot
plt.grid()
plt.plot(t1, f(t1), 'b-')

plt.subplot(212) # 2 --> rows. 1 -- column. 1 --> 2nd subplot
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.savefig("0.4_multiple_plot.png")

plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
plt.savefig("0.4_multiple_plot2.png")




