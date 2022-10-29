def pi():
    true_pi = 3.141593
    estimated_pi = 0
    n = 1
    m = 1

    while round(4*estimated_pi, 6) != true_pi:
        estimated_pi += (m/n)
        n += 2
        m *= -1
        print(4*estimated_pi)

pi()
