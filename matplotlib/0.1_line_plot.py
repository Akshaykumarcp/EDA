import matplotlib.pyplot as plt

# specify only y-axis
plt.plot([1,2,3,4])
plt.ylabel("y-axis Numbers")
plt.xlabel("x-axis Numbers")
plt.title("Title - Numbers")
plt.savefig("0.1_line_plot.png")
# then x-axis is 0,1,2,3.. so on

#  in above plot, there is not grid in background, lets turn on grid option
plt.plot([1,2,3,4],[1,4,9,16]) # 1st list is the x-axis. 2nd list is the y-axis
plt.ylabel("Squares")
plt.xlabel("Numbers")
plt.title("Squares function")
plt.grid()
plt.savefig("0.1_line_plot_with_grid.png")

