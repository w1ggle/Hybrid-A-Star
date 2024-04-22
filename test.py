import math
states = [(1,2,3),(1,2,3)]

momentumFactor = 2
velocity = 1

for i in range(states - 1):
    distance = math.hypot(x[i+1] - x[i], y[i+1] - y[i])
    time = distance / (velocity * momentumFactor)