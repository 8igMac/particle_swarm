# Particle swarm optimization on a function.
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import math

# env parameters
map_margin = 10
max_velocity = 0.07
num_particle = 100
num_one_side_neighbor = 5

# Velocity update parameters
# c0 = 0.65  # inertia coefficient
# c1 = 3.7  # personal_best coefficient
# c2 = 0.3  # local_best coefficient

# slow convergence
# c0 = 0.8  # inertia coefficient
# c1 = 3.3  # personal_best coefficient
# c2 = 0.7  # local_best coefficient

# c0 = 0.8  # inertia coefficient
# c1 = 2  # personal_best coefficient
# c2 = 0.5  # local_best coefficient

c0 = 1  # inertia coefficient
c1 = 2  # personal_best coefficient
c2 = 2  # local_best coefficient

# Target function parameters.
x0 = 0.2
y0 = 0.3
r = 1
a = 8
b = 1
c = 5 
d = 2

class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.fitness = -1

class Velocity:
    pass

class Particle:
    def __init__(self):
        # Init random position (limit on map margin).
        self.position = Position()
        self.position.x = random.uniform(-map_margin, map_margin)
        self.position.y = random.uniform(-map_margin, map_margin)
        self.position.fitness = eval_fitness(
                self.position.x, self.position.y)
        # Init random velocity.
        self.velocity = Velocity()
        self.velocity.x = random.uniform(-max_velocity, max_velocity)
        self.velocity.y = random.uniform(-max_velocity, max_velocity)
        # Init personal best
        self.personal_best = self.position

    # Update position and personal best
    # note: did not limit particles position on map margin
    def update_position(self):
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        self.position.fitness = eval_fitness(
                self.position.x, self.position.y)
        
        if self.position.fitness > self.personal_best.fitness:
            self.personal_best = self.position

    # Update velocity based on personal and local best.
    def update_velocity(self, local_best):
        new_vx = \
            c0*self.velocity.x \
          + c1*(self.personal_best.x - self.position.x) \
          + c2*(local_best.x - self.position.x)
        # Check maximun velocity bound
        if abs(new_vx) > max_velocity:
            self.velocity.x = new_vx/abs(new_vx) * max_velocity
        else:
            self.velocity.x = new_vx

        new_vy = \
            c0*self.velocity.y \
          + c1*(self.personal_best.y - self.position.y) \
          + c2*(local_best.y - self.position.y)
        # Check maximun velocity bound
        if abs(new_vy) > max_velocity:
            self.velocity.y = new_vy/abs(new_vy) * max_velocity
        else:
            self.velocity.y = new_vy

# Target function.
def eval_fitness(x, y):
    # Origin function in the spec
    # return \
    #     np.exp(-((x - x0)**2 + (y - y0)**2) / r**2) \
    #   * np.sin(a*x)**2 \
    #   * np.sin(b*x + c*y + d*x**2)**2

    # Some tuning of origin function
    return \
        np.exp(-((x - x0)**2 + (y - y0)**2) / r**2) \
      * np.sin(a*x)**2 \
      * np.sin(b*x)**2 \
      * np.sin(b*x + c*y**2 + d*x**3)**8

    # Rastrigin function
    # return -( \
    #     10*2 \
    #   + (x**2 - 20*np.cos(2*math.pi*x)) \
    #   + (y**2 - 20*np.cos(2*math.pi*y)) )

    # Easom function
    # return ( \
    #     np.cos(x) \
    #   * np.cos(y) \
    #   * np.exp(-((x-math.pi)**2 + (y-math.pi)**2)) \
    # )

    # Tree-hump camel function
    # return 0.0001 * ( np.abs(\
    #     np.sin(x) * np.sin(y) * np.exp(np.abs(\
    #         100 - np.sqrt(x**2 + y**2)/math.pi))\
    #     ) + 1) ** 0.1


def get_global_best(group):
    best = Position()
    for i in group:
        if i.personal_best.fitness > best.fitness:
            best = i.position
    return best

# Get neighborhood best. Implement virtual circle neighborhood.
def get_local_best(group, index):
    best = Position()
    for i in range(num_one_side_neighbor):
        if group[(index + (i+1)) % len(group)].personal_best.fitness > best.fitness:
            best = group[(index + (i+1)) % len(group)].position
        if group[(index - (i+1)) % len(group)].personal_best.fitness > best.fitness:
            best = group[(index - (i+1)) % len(group)].position
    return best

# Create a group of particles.
group = [Particle() for i in range(num_particle)]

# Plot background.
delta = 0.005
x = np.arange(-map_margin, map_margin, delta)
y = np.arange(-map_margin, map_margin, delta)
X, Y = np.meshgrid(x, y)
Z = eval_fitness(X, Y)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Particle Swarm Optimization')
fig.colorbar(CS, shrink=0.8, extend='both')

# Plot particle
x = [i.position.x for i in group]
y = [i.position.y for i in group]
line, = ax.plot(x, y, 'ro', markersize=4)

# Run episode
def run(episode):
    print(episode)
    for i in group:
        i.update_position()
    for i, v in enumerate(group):
        local_best = get_local_best(group, i)
        v.update_velocity(local_best)

    # Update plot
    for i in range(len(group)):
        x[i] = group[i].position.x
        y[i] = group[i].position.y

    line.set_data(x, y)
    return line

ani = animation.FuncAnimation(fig, run)
plt.show()
