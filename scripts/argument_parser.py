def fib(nums):
    x, y = 0, 1
    for _ in range(nums):
        x, y = y, x+y
        # print(x)
        yield x
        # print(x)
        
def sum_1(num):
    for i in num:
        print(i, i**2)
        yield i**2
    
for i in fib(10):
    print(i, sum_1(i))
# print(next(fib(10)))

# print(fib(10))