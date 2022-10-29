def convert_to_hex(n):
    if n == 0:
        return 0
    
    hexes = {10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F'}

    current = n
    converted_num = []

    while current != 0:
        new_num = current % 16
        if new_num >= 10:
            new_num = hexes[new_num]
        converted_num.append(str(new_num))
        current = current // 16

    converted_num.reverse()
    return ''.join(converted_num)

print(convert_to_hex(100)) # expect 64
print(convert_to_hex(15)) # expect F
print(convert_to_hex(29)) # expect 1D
