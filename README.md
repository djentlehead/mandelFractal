# Mandelbrot Set Visualizer (CUDA-Accelerated)

This is a high-performance visualizer for the Mandelbrot set built using Python, Pygame for the graphical interface, and CUDA (via Numba) for GPU-accelerated computation.

## 🚀 Features

- **Real-time Mandelbrot rendering** with GPU acceleration via Numba CUDA
- **Interactive zooming** using the mouse scroll wheel
- **Color-mapped rendering** based on iteration depth
- Efficient performance even on high-resolution screens

---

## 🧠 What is the Mandelbrot Set?

The Mandelbrot set is a famous fractal defined in the complex plane. It represents a set of complex numbers for which the function *f(z) = z² + c* does not diverge when iterated from *z = 0*. This project visually renders this beautiful fractal pattern.

---

## 📸 Preview

![image](https://github.com/user-attachments/assets/312e0d0d-ca95-4286-981f-688baf1b0965)

---

## 🛠️ Installation

### Prerequisites

Ensure the following are installed:

- Python 3.8+
- CUDA-enabled GPU
- [Numba](https://numba.pydata.org/)
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)

You can install the required Python packages using:

```bash
pip install pygame numpy numba
