import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(filename, title, output_image):
    data = np.loadtxt(filename, dtype=int)

    plt.figure(figsize=(8, 4))
    plt.bar(range(256), data, color='black', width=1.0)
    plt.title(title)
    plt.xlabel("Intensità (0-255)")
    plt.ylabel("Frequenza")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()

plot_histogram("histogram_before.txt", "Istogramma Prima dell'Equalizzazione", "histogram_before.png")
plot_histogram("histogram_after.txt", "Istogramma Dopo l'Equalizzazione", "histogram_after.png")

print("Python file executed successfully!")