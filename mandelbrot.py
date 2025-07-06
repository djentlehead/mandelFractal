# pylint: disable=import-error
import pygame
import numpy as np
import time
from numba import cuda

pygame.init()

width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Set Visualizer")

min_x, max_x = -2.0, 1.0
min_y, max_y = -1.5, 1.5

def get_max_iter():
    zoom_level = abs(max_x - min_x)
    return int(100 + 50 * (1 / zoom_level))

@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, image, max_iter):
    height, width = image.shape
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    real = min_x + x * (max_x - min_x) / width
    imag = min_y + y * (max_y - min_y) / height
    c_real = real
    c_imag = imag
    z_real = 0.0
    z_imag = 0.0
    count = 0

    while z_real * z_real + z_imag * z_imag <= 4.0 and count < max_iter:
        temp = z_real * z_real - z_imag * z_imag + c_real
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = temp
        count += 1

    image[y, x] = count

def draw_mandelbrot():
    max_iter = get_max_iter()
    image = np.zeros((height, width), dtype=np.uint16)
    d_image = cuda.to_device(image)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, d_image, max_iter)
    d_image.copy_to_host(image)

    surface = pygame.Surface((width, height))
    pixels = pygame.surfarray.pixels3d(surface)
    for y in range(height):
        for x in range(width):
            m = image[y, x]
            color = (m * 9 % 255, m * 7 % 255, m * 5 % 255)
            pixels[x, y] = color
    del pixels
    screen.blit(surface, (0, 0))
    pygame.display.flip()

def zoom(factor, mouse_pos):
    global min_x, max_x, min_y, max_y
    mouse_x, mouse_y = mouse_pos
    cx = min_x + (mouse_x / width) * (max_x - min_x)
    cy = min_y + (mouse_y / height) * (max_y - min_y)

    new_width = (max_x - min_x) * factor
    new_height = (max_y - min_y) * factor

    min_x = cx - new_width / 2
    max_x = cx + new_width / 2
    min_y = cy - new_height / 2
    max_y = cy + new_height / 2

needs_redraw = True
last_zoom_time = 0
zoom_cooldown = 0.2  

draw_mandelbrot()
needs_redraw = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            now = time.time()
            if now - last_zoom_time > zoom_cooldown:
                if event.button == 4:  
                    zoom(0.5, event.pos)
                    needs_redraw = True
                elif event.button == 5:  
                    zoom(2.0, event.pos)
                    needs_redraw = True
                last_zoom_time = now

    if needs_redraw:
        draw_mandelbrot()
        needs_redraw = False

pygame.quit()


