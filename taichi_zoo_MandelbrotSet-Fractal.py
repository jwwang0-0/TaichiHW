import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.Vector.field(3, dtype=float, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        # c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        # z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        c = ti.Vector([i / n - 1, j / n - 0.5]) * 3
        z = ti.Vector([0.0, 0.0])
        iterations = 0
        while z.norm() < 2 and iterations < (t%50):
            z = complex_sqr(z) + c
            iterations += 1
        #print(z.norm())
        pixels[i, j] = [0.2,(1 - iterations * 0.02),0.9]


gui = ti.GUI("Mandelbrot Set", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.1)
    gui.set_image(pixels)
    gui.show()
