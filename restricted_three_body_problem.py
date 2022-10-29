from vpython import *

# simulate the motion of the earth around the sun and the moon around the earth
    
# define the gravitational constant
G = 6.67408e-11

# mass of the earth, moon, and sun
m_earth = 5.973e24
m_moon = 7.347e22
m_sun = 1.989e30

# distance between earth and moon, and sun and earth
distance_EM = 3.844e8
distance_SE = 1.496e11

# force of gravity between earth and moon, and sun and earth
F_grav_EM = G*(m_earth*m_moon)/(distance_EM**2)
F_grav_SE = G*(m_sun*m_earth)/(distance_SE**2)

# angular velocity of the moon and the earth  (rad/s)
velocity_moon = (F_grav_EM/(m_moon * distance_EM))**0.5
velocity_earth = (F_grav_SE/(m_earth * distance_SE))**0.5

# initial velocity of moon
v = vector(0.5,0,0)

# earth, moon, and sun objects (radii not to scale)
earth = sphere(pos=vector(3,0,0),color=color.blue,radius=.3,make_trail=True)
moon = sphere(pos=earth.pos+v,color=color.white,radius=0.1,make_trail=True)
sun = sphere(pos=vector(0,0,0),color=color.yellow,radius=1)

t = 0
dt = 3600

while t < 1e10:
    rate(100)

    angle_earth = velocity_earth * dt
    angle_moon = velocity_moon * dt
    
    earth.pos = rotate(earth.pos,angle=angle_earth)
    v = rotate(v,angle=angle_moon)
    moon.pos = earth.pos + v
    
    t += dt
