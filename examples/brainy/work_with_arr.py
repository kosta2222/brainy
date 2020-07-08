from .util import _0_
def copy_vector(src:list,dest:list,n):
    len_=0
    for i in range(n):
        dest[i] =\
            src[i]
        len_+=1
    return len_

def to_ribbon(src, dest, in_,out):
      len_=0
      for row in range(out):
         for elem in range(in_):
             dest[row * in_ + elem] = src[row][elem]
             len_+=1
      return len_
    # _0_("copy_matrixAsStaticSquare_toRibon");
