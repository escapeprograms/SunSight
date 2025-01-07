TEST_VAR = 4

def change(x):
    global TEST_VAR
    TEST_VAR = x
    print("new var", TEST_VAR)

def read():
    global TEST_VAR
    print("cur var", TEST_VAR)


read()
change(12)
read()