# given n > 1, return n**(1/3)

def cubed_root(n):
    a = n
    b = 1
    m = (a + b) / 2
    
    while round(m*m*m, 4) != n:
        if m*m*m < n:
            b = m
        else:
            a = m
        m = (a + b) / 2
        
    return round(m, 4)

# testing vs built-in function

print(cubed_root(8) == round(8**(1/3), 4))
print(cubed_root(3890) == round(3890**(1/3), 4))
print(cubed_root(2.1) == round(2.1**(1/3), 4))

# given n > 1, return n**0.5

def square_root(n):
    old_guess = n / 2
    new_guess = n
    while abs(new_guess - old_guess) > 0.000001:
        old_guess = new_guess
        new_guess = (( n / old_guess) + old_guess) / 2
        print(new_guess, abs(new_guess - old_guess))
    return

square_root(9)
