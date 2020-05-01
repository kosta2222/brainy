from util_func import _0_
def copy_vector(src:list,dest:list,n):
    for i in range(n):
        dest[i] =\
            src[i]
def copy_matrixAsStaticSquare_toRibon(src, dest, in_,out):
    for row in range(out):
        for elem in range(in_):
            dest[row * in_ + elem] = src[row][elem];
    _0_("copy_matrixAsStaticSquare_toRibon");
