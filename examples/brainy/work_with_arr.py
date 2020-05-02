from .util_func import _0_
def copy_vector(src:list,dest:list,n):
    for i in range(n):
        dest[i] =\
            src[i]
def copy_matrixAsStaticSquare_toRibon(src, dest, in_,out):
    try:
      print("out",out)
      for row in range(out):
         for elem in range(in_):
             dest[row * in_ + elem] = src[row][elem]
    except Exception:
        print("Exc")
        print("out",out)
    _0_("copy_matrixAsStaticSquare_toRibon");
