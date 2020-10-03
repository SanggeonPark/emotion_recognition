# Switcher : from label string to integer
def neutral():
    return 0
def anger():
    return 1
def surprise():
    return 2
def smile():
    return 3
def sad():
    return 4

l2i_switcher = {
    'neutral': neutral,
    'anger': anger,
    'surprise': surprise,
    'smile': smile,
    'sad': sad
}

# Switcher : from  integer to string
def zero():
    return 'neutral'
def one():
    return 'anger'
def two():
    return 'surprise'
def three():
    return 'smile'
def four():
    return 'sad'

i2l_switcher = {
    0: zero,
    1: one,
    2: two,
    3: three,
    4: four
}
