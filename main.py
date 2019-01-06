# Particle swarm optimization on a function.
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import math

# env parameters
margin = 4
max_velocity = 0.01
num_particle = 20
max_episode = 30

# Velocity update parameters
c0 = 0.5  # inertia coefficient
c1 = 1  # personal_best coefficient
c2 = 0.5  # global_best coefficient

# Target function parameters.
# x0 = 0.2
# y0 = 0.3
# r = 1
# a = 8
# b = 1
# c = 5 
# d = 2
x0 = 0.2
y0 = 0.3
r = 2
a = 8
b = 8
c = 8 
d = 8

class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.fitness = -1

class Velocity:
    pass

class Particle:
    def __init__(self):
        # Init random position. 
        self.position = Position()
        self.position.x = random.uniform(-margin, margin)
        self.position.y = random.uniform(-margin, margin)
        self.position.fitness = eval_fitness(
                self.position.x, self.position.y)
        # Init random velocity.
        self.velocity = Velocity()
        self.velocity.x = random.uniform(-max_velocity, max_velocity)
        self.velocity.y = random.uniform(-max_velocity, max_velocity)
        # Init personal best
        self.personal_best = self.position

    # Update position and personal best
    def update_position(self):
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        # debug
        # self.position.x += 0.01
        # self.position.y += 0.01
        self.position.fitness = eval_fitness(
                self.position.x, self.position.y)
        
        if self.position.fitness > self.personal_best.fitness:
            self.personal_best = self.position

    # Update velocity based on personal and global best.
    def update_velocity(self, global_best):
        self.velocity.x = \
            c0*self.velocity.x \
          + c1*(self.personal_best.x - self.position.x) \
          + c2*(global_best.x - self.position.x)

        self.velocity.y = \
            c0*self.velocity.y \
          + c1*(self.personal_best.y - self.position.y) \
          + c2*(global_best.y - self.position.y)

def eval_fitness(x, y):
    return \
        math.exp(-((x - x0)**2 + (y - y0)**2) / r**2) \
      * math.sin(a*x)**2 \
      * math.sin(b*x + c*y + d*x**2)**2

def get_global_best(group):
    best = Position()
    for i in group:
        if i.personal_best.fitness > best.fitness:
            best = i.position
    return best



# Create a group of particles.
group = [Particle() for i in range(num_particle)]

# Plot background.
delta = 0.005
x = np.arange(-margin, margin, delta)
y = np.arange(-margin, margin, delta)
X, Y = np.meshgrid(x, y)
Z = np.exp(-((X - x0)**2 + (Y - y0)**2) / r**2) \
    * np.sin(a*X)**2 \
    * np.sin(b*X + c*Y + d*X**2)**2

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.set_title('Particle Swarm Optimization')
fig.colorbar(CS, shrink=0.8, extend='both')

# Plot particle
x = [i.position.x for i in group]
y = [i.position.y for i in group]
line, = ax.plot(x, y, 'ro', markersize=4)

def run(episode):
    print(episode)
    for i in group:
        i.update_position()
    global_best = get_global_best(group)
    for i in group:
        i.update_velocity(global_best)

    # Update plot
    for i in range(len(group)):
        x[i] = group[i].position.x
        y[i] = group[i].position.y

    line.set_data(x, y)
    return line

ani = animation.FuncAnimation(fig, run, max_episode)
plt.show()
