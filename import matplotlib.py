import matplotlib.pyplot as plt
iteration_numbers = iteration_numbers = [20, 50, 100, 200, 500, 1000, 2000]

aco_eas_results = [41253, 37968, 37192, 25327, 25003, 25109, 25004]
aco_mmas_results = [27835, 26902, 25020, 25132, 25705, 25097, 24914]
ga_adaptive_results = [68253, 70434, 52580, 41899, 31144, 27103, 25431]
ga_results = [73253, 71434, 56580, 48899, 34144, 28203, 25831]

plt.plot(iteration_numbers, aco_eas_results, label="ACO EAS", marker='o', color='blue')
plt.plot(iteration_numbers, aco_mmas_results, label="ACO MMAS", marker='o', color='green')
plt.plot(iteration_numbers, ga_adaptive_results, label="GA Adaptive Hybrid", marker='o', color='red')
plt.plot(iteration_numbers, ga_results, label="Non-Adaptive Hybrid", marker='o', color='orange')
plt.xlabel("Number of Iterations")
plt.ylabel("Average Best Tour Length")
plt.title("Comparison of ACO EAS, ACO MMAS, and GA Adaptive Hybrid")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plot.png")
plt.show()