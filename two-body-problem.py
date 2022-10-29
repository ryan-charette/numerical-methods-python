from vpython import *

def main():
    # simulate the motion of the earth around the sun
    
    # define the gravitational constant
    G = 6.67408e-11
    
    # mass of the sun
    m_sun = 1.9891e30 # kg
    
    # radius of the sun (not to scale)
    # this is so earth is more easily visible in the simulation
    # r_sun = 6.9634e8 # (actual)
    r_sun = 5e10 # (simulated)
    
    # mass of the earth
    m_earth = 5.972e24
    
    # radius of the earth (not to scale)
    # r_earth = 6.367e6 # (actual)
    r_earth = 1e10 # (simulated)
    
    # create a sun object
    sun = sphere(pos = vector(0,0,0), radius = r_sun, color = color.yellow)
    
    # create an earth object
    earth = sphere(pos = vector(1.496e11, 0, 0), radius = r_earth, color = color.blue, make_trail = True)
    
    # initialize the earth's velocity
    earth.velocity = vector(0, 3e4, 0)
    
    # define initial time
    t = 0
    
    # define time interval
    dt = 3600
    
    # simulate the motion
    while (t < 1e10):
        rate(300)
        earth.pos = earth.pos + earth.velocity * dt
        r_vector = earth.pos - sun.pos
        F_grav = (-(G * m_sun * m_earth) / (mag(r_vector)**2)) * (norm(r_vector))
        earth.velocity = earth.velocity + (F_grav / m_earth) * dt
        t = t + dt
        
main()
