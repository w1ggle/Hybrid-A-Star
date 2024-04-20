states = [(1,2,3),(1,2,3)]

x , y, yaw = 0

for i in range(len(x) - 1):
    if yaw(i) == yaw(i+1):
        break
    elif yaw(i) >= yaw(i+1):
        print("turn right")
    else:
        print("turn left")
        
    if x(i) == x(i+1):
        break
    elif x(i) >= x(i+1):
        print("turn right")
    else:
        print("turn left")