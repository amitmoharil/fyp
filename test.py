def fib_gen(n):

  a = 0 
  b = 1 
  yield a 
  yield b
  for i in range(n):
    yield a+b 
    temp = b
    b = a+b 
    a = temp 
  
for x in fib_gen(10):
  print(x)