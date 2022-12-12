import matplotlib.pyplot as plt

# Draw the finite difference stencil
plt.subplot(1, 2, 1)
plt.title("Time Step $t_n$")

plt.plot(0, -1, "bo")
plt.text(0, -1, "$U_{i,j-1}^n$")

plt.plot(-1, 0, "bo")
plt.text(-1, 0, "$U_{i-1,j}^n$")

plt.plot(0, 0, "bo")
plt.text(0, 0, "$U_{i,j}^n$")

plt.plot(1, 0, "bo")
plt.text(1, 0, "$U_{i+1,j}^n$")

plt.plot(0, 1, "bo")
plt.text(0, 1, "$U_{i,j+1}^n$")

plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

plt.grid()


plt.subplot(1, 2, 2)
plt.title("Time Step $t_{n+1}$")

plt.plot(0, 0, "ro")
plt.text(0, 0, "$U_{i,j}^{n+1}$")

plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])

plt.grid()

plt.tight_layout()

plt.show()
