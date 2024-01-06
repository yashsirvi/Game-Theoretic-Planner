import time
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import matplotlib.cm as cm
from scipy.interpolate import CubicSpline

control_points = [[0, 1], [1, 1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2, 0], [0, 1]]
t = np.array(range(len(control_points)))
t_rad = np.linspace(0, 2*np.pi, 15)
colors = cm.rainbow(np.linspace(0, 1, 15))
circle_points = [2.3*np.cos(t_rad), 2.3*np.sin(t_rad)]
circle_points_x = np.flip(np.roll(circle_points[0], -4))
circle_points_y = np.flip(np.roll(circle_points[1], -4))
cs = CubicSpline(t, control_points, bc_type='periodic')
coeffs = cs.c

third_powers = coeffs[3]
second_powers = coeffs[2]
first_powers = coeffs[1]
no_powers = coeffs[0]

k = 4
x_1_3 = third_powers[k][0]
x_1_2 = second_powers[k][0]
x_1_1 = first_powers[k][0]
x_1_0 = no_powers[k][0]

y_1_3 = third_powers[k][1]
y_1_2 = second_powers[k][1]
y_1_1 = first_powers[k][1]
y_1_0 = no_powers[k][1]

test_points = cs(np.linspace(0,len(control_points)))
test_points_2 = np.linspace(0 ,1)
t = ca.MX.sym('t')
P = ca.MX.sym('P', 10) #all the parameters 

Z = ca.blockcat([[P[2]*t**3 + P[3]*t**2 + P[4]*t + P[5]], [P[6]*t**3 + P[7]*t**2 + P[8]*t + P[9]]])
curve = ca.Function('curve', [t], [Z])
distance = ca.norm_2(P[:2] - curve(t))
opts ={}
opts['ipopt.print_level'] = 0
# opts['jit'] = True
# curve.print_options()
# opts['akjdfhasd'] = True
#opts['verbose'] = True

nlp = {'x': t, 'f': distance, 'p': P}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

def find_closest_point(params):
    
    sol = solver(x0=0.5, lbx = 0, ubx = 1, p=params)
    t_opt = sol['x']
    t_dis = sol['f']

    print(t_opt)

    return t_opt, t_dis

ans, dist = find_closest_point(np.array([-1.5, 1.5, x_1_0, x_1_1, x_1_2, x_1_3, y_1_0, y_1_1, y_1_2, y_1_3]))
print(dist)
# x = cs(k+ans)[0][0][0]
# y = cs(k+ans)[0][0][1]

ans = float(ans)
print(ans)
x = x_1_0*ans**3 + x_1_1*ans**2 + x_1_2*ans + x_1_3
y = y_1_0*ans**3 + y_1_1*ans**2 + y_1_2*ans + y_1_3

# plt.plot(x, y, 'ro')
# plt.plot([-1.5],[1.5], 'ro')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot([i[0] for i in test_points], [i[1] for i in test_points])
plt.scatter(circle_points_x, circle_points_y, c = colors)
#OK SO this is how you extract the coefficients of the spline piecewise polynomial
#plt.plot([x_1_0*i**3 + x_1_1*i**2 + x_1_2*i + x_1_3 for i in test_points_2], [y_1_0*i**3 + y_1_1*i**2 + y_1_2*i + y_1_3 for i in test_points_2], 'g--')

k = 0
t_val = 0
n = len(circle_points_x)
yy = len(control_points)
e = 0.1
# circle_points_x = np.flip(circle_points_x)
# circle_points_y = np.flip(circle_points_y)
print([[i,j] for i,j in zip(circle_points_x,circle_points_y)])

time1 = time.time()

for i in range(n):
    k = k%(yy-1)
    x03 = third_powers[k][0]
    x02 = second_powers[k][0]
    x01 = first_powers[k][0]
    x00 = no_powers[k][0]

    y03 = third_powers[k][1]
    y02 = second_powers[k][1]
    y01 = first_powers[k][1]
    y00 = no_powers[k][1]

    ans0, dist0 = find_closest_point(np.array([circle_points_x[i], circle_points_y[i], x00, x01, x02, x03, y00, y01, y02, y03]))

    nk = (k+1)%(yy-1)
    x13 = third_powers[nk][0]
    x12 = second_powers[nk][0]
    x11 = first_powers[nk][0]
    x10 = no_powers[nk][0]

    y13 = third_powers[nk][1]
    y12 = second_powers[nk][1]
    y11 = first_powers[nk][1]
    y10 = no_powers[nk][1]

    ans1, dist1 = find_closest_point(np.array([circle_points_x[i], circle_points_y[i], x10, x11, x12, x13, y10, y11, y12, y13]))

    nkk = (k+2)%(yy-1)
    x23 = third_powers[nkk][0]
    x22 = second_powers[nkk][0]
    x21 = first_powers[nkk][0]
    x20 = no_powers[nkk][0]

    y23 = third_powers[nkk][1]
    y22 = second_powers[nkk][1]
    y21 = first_powers[nkk][1]
    y20 = no_powers[nkk][1]

    ans2, dist2 = find_closest_point(np.array([circle_points_x[i], circle_points_y[i], x20, x21, x22, x23, y20, y21, y22, y23]))

    # nkkk = (k+3)%(yy-1)
    # x33 = third_powers[nkk][0]
    # x32 = second_powers[nkk][0]
    # x31 = first_powers[nkk][0]
    # x30 = no_powers[nkk][0]

    # y33 = third_powers[nkk][1]
    # y32 = second_powers[nkk][1]
    # y31 = first_powers[nkk][1]
    # y30 = no_powers[nkk][1]

    # ans3, dist3 = find_closest_point(np.array([circle_points_x[i], circle_points_y[i], x30, x31, x32, x33, y30, y31, y32, y33]))

    # if dist < dist2 and dist3 < dist1 and dist3 < dist0:
    #     k = nkkk
    #     t_val = nkkk + ans3
    #     x = cs(t_val)[0][0][0]
    #     y = cs(t_val)[0][0][1]
    #     #plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')

    if dist2 < dist1 and dist2 < dist0:
        k = nkk
        t_val = nkk + ans2
        x = cs(t_val)[0][0][0]
        y = cs(t_val)[0][0][1]
        #plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')
    elif dist1 < dist0:
        k = nk
        t_val = nk + ans1
        x = cs(t_val)[0][0][0]
        y = cs(t_val)[0][0][1]
        #plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')
    else:
        t_val = k + ans0 
        x = cs(t_val)[0][0][0]
        y = cs(t_val)[0][0][1]
        #plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')

time2 = time.time()
print(time2-time1)
    

#plt.plot([x_1_0*i**3 + x_1_1*i**2 + x_1_2*i + x_1_3 for i in test_points_2], [y_1_0*i**3 + y_1_1*i**2 + y_1_2*i + y_1_3 for i in test_points_2], 'r--')

#plt.show()