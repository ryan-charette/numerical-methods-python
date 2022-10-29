#   Consider a function defined by the following points:
#   (1.00, 343), (3.25, 568), (5.50, 645), (7.75, 115), (10.00, 134)
#   Create a difference table for this function.

x0 = 1.00
x1 = 3.25
x2 = 5.50
x3 = 7.75
x4 = 10.00

y0 = 343
y1 = 568
y2 = 645
y3 = 115
y4 = 134

dy0 = y1 - y0
dy1 = y2 - y1
dy2 = y3 - y2
dy3 = y4 - y3

d2y0 = dy1 - dy0
d2y1 = dy2 - dy1
d2y2 = dy3 - dy2

d3y0 = d2y1 - d2y0
d3y1 = d2y2 - d2y1

d4y0 = d3y1 - d3y0

#   a) Use Newton's forward interpolation formula to obtain the value
#      of the function at x = 2.

x = 2
u = (x - x0) / (x1 - x0)

fx = (y0) + (u * dy0) + ((u*(u - 1) / 2) * d2y0) + ((u*(u-1)*(u-2) / 6) * d3y0) + ((u*(u-1)*(u-2)*(u-3) / 24) * d4y0)
print(fx)

#   b) Use Newton's backward interpolation formula to obtain the value
#      of the function at x = 9.

x = 9
u = (x - x4) / (x1 - x0)
print(u)

fx = (y4) + (u * dy3) + ((u*(u + 1) / 2) * d2y2) + ((u*(u + 1)*(u + 2) / 6) * d3y1) + ((u*(u + 1)*(u + 2)*(u + 3) / 24) * d4y0)
print(fx)

#   c) Use Stirling's central difference formula to obtain the value of
#      the function at x = 7.

x0 = 2
x1 = 4
x2 = 6
x3 = 8
x4 = 10

y0 = 5
y1 = 49
y2 = 181
y3 = 449
y4 = 901

dy0 = y1 - y0
dy1 = y2 - y1
dy2 = y3 - y2
dy3 = y4 - y3

d2y0 = dy1 - dy0
d2y1 = dy2 - dy1
d2y2 = dy3 - dy2

d3y0 = d2y1 - d2y0
d3y1 = d2y2 - d2y1

d4y0 = d3y1 - d3y0

x = 7
u = (7 - x2) / (x1 - x0)

fx1 = (y2) + (u * dy0) + ((u*(u - 1) / 2) * d2y0) + ((u*(u-1)*(u-2) / 6) * d3y0) + ((u*(u-1)*(u-2)*(u-3) / 24) * d4y0)
fx2 = (y2) + (u * dy3) + ((u*(u + 1) / 2) * d2y2) + ((u*(u + 1)*(u + 2) / 6) * d3y1) + ((u*(u + 1)*(u + 2)*(u + 3) / 24) * d4y0)
print((fx1 + fx2) / 2)
