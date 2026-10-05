import numpy as np
import matplotlib.pyplot as plt

radius = 1.0                    
canvas = 3.0                  
N = 500                      
albedo, light_intensity = 1.0, 1.0         

xs = np.linspace(-canvas/2, canvas/2, N)
ys = np.linspace(-canvas/2, canvas/2, N)

def create_sphere(xs, ys, radius):
    X, Y = np.meshgrid(xs, ys)
    r2 = X**2 + Y**2
    mask = r2 <= radius**2
    Z = np.sqrt(np.maximum(radius**2 - r2, 0))
    return X, Y, Z, mask

X, Y, Z, mask = create_sphere(xs, ys, radius)

n = np.stack([X, Y, Z], axis=-1) / radius

def normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

def lambertian(n, L):
    return albedo * light_intensity * np.maximum(np.dot(n, L), 0)

def phong(n, L, V, exponent):
    nl = np.maximum(np.dot(n, L), 0)[..., None]
    r = 2 * nl * n - L
    view = np.maximum(np.dot(r, V), 0) ** exponent
    return albedo * light_intensity * view * (nl[..., 0] > 0)       

def render(fn):
    img = np.zeros((N, N))
    img[mask] = fn(n[mask])
    return img

V = normalize([1, 0, 0])  # view direction

settings = {
    "Light = camera":   normalize([0, 0, 1]),
    "Light from upper right":     normalize([1, 1, 1]),
    "Light from upper left":    normalize([-1, 1, 1]),
    "Lighting from above":     normalize([0, 0, 1]),
    "Light from left":   normalize([-1, 0, 0.5]),
}

exponents = [1, 5, 25, 100]

fig, axes = plt.subplots(len(settings), 1 + len(exponents), figsize=(14, 8))
for i, (name, L) in enumerate(settings.items()):
    ax = axes[i, 0]
    ax.imshow(render(lambda nn: lambertian(nn, L)), cmap="gray", vmin=0, vmax=1,
              extent=[-canvas/2, canvas/2, -canvas/2, canvas/2], origin="lower")
    ax.set_title("Lambertian" if i == 0 else "")
    ax.set_ylabel(name, fontsize=8)
    for j, p in enumerate(exponents, start=1):
        ax = axes[i, j]
        ax.imshow(render(lambda nn: phong(nn, L, V, p)), cmap="gray", vmin=0, vmax=1,
                  extent=[-canvas/2, canvas/2, -canvas/2, canvas/2], origin="lower")
        if i == 0:
            ax.set_title(f"Phong, n = {p}")

plt.show()