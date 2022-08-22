# n is a number in base 10
# b is a new base
# returns n in base b
def convert(n, b):

    if n == 0:
        return 0

    current = n
    converted_num = []

    while current != 0:
        converted_num.append(str(current % 2))
        current = current // 2

    converted_num.reverse()
    return ''.join(converted_num)

def test_case():
    print(convert(43, 2))

test_case()