#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

using Float = float;

struct dim2 {
    size_t x = 0;
    size_t y = 0;
};

void BlockMul(const Float* a, const Float* b, Float* c, dim2 blockDimA, dim2 blockDimB, dim2 dimA, dim2 dimB,
    dim2 offsetA, dim2 offsetB, dim2 offsetC)
{
    const size_t nAcols = dimA.x;
    const size_t nArows = dimA.y;
    const size_t nBcols = dimB.x;
    const size_t nBrows = dimB.y;
    const size_t nABlockRows = blockDimA.y;
    const size_t nABlockCols = blockDimB.x;
    const size_t nBBlockCols = blockDimB.x;
    const size_t nCcols = nBcols;
    const size_t nCrows = nArows;
    assert(nAcols == nBrows);
    for (size_t row = 0; row != nABlockRows; ++row) {
        for (size_t col = 0; col != nBBlockCols; ++col) {
            Float v = Float(0);
            for (size_t c = 0; c != nABlockCols; ++c) {
                v += a[(row + offsetA.y) * nAcols + offsetA.x + c] * b[(c + offsetB.y) * nBcols + offsetB.x + col];
            }
            c[(row + offsetC.y) * nCcols + offsetC.x + col] = v;
        }
    }
}

void InplaceMatAdd(const Float* a, Float* c, dim2 dimA, dim2 dimC, dim2 blockDimA, dim2 offsetA, dim2 offsetC)
{
    const size_t nAcols = dimA.x;
    const size_t nArows = dimA.y;
    const size_t nABlockRows = blockDimA.y;
    const size_t nABlockCols = blockDimA.x;
    const size_t nCcols = dimC.x;
    const size_t nCrows = dimC.y;
    for (size_t row = 0; row != nABlockRows; ++row) {
        for (size_t col = 0; col != nABlockCols; ++col) {
            c[(offsetC.y + row) * nCcols + offsetC.x + col] += a[(offsetA.y + row) * nAcols + offsetA.x + col];
        }
    }
}

void MatMul(const Float* a, const Float* b, Float* c, Float* block, dim2 blockDim, dim2 dimA, dim2 dimB)
{
    const size_t nAcols = dimA.x / blockDim.x;
    const size_t nArows = dimA.y / blockDim.y;
    const size_t nBcols = dimB.x;
    const size_t nBrows = dimB.y;
    const size_t nCcols = nBcols;
    const size_t nCrows = nArows;
    const dim2 dimC = {dimB.x, dimA.y};
    for (size_t row = 0; row != nArows; ++row) {
        for (size_t col = 0; col != nBcols; ++col) {
            for(size_t y = 0; y != nAcols; ++y ) {
              //C[row][col] = A[row][c] x B[c][col];
              const dim2 offsetA = {y * blockDim.x, row * blockDim.y};
              const dim2 offsetB = {col * blockDim.x, y * blockDim.y};
              const dim2 offsetBlock = {0, 0};
              const dim2 offsetC = {col * blockDim.x, row * blockDim.y};
              BlockMul(a, b, block, blockDim, blockDim, dimA, dimB, offsetA, offsetB, offsetC);
              InplaceMatAdd(block, c, dimA, dimC, blockDim, {0,0}, offsetC);
            }
        }
    }
}

void Print(const dim2& size, const Float* m) {
  for(size_t row = 0; row != size.y; ++row) {
    for(size_t col = 0; col != size.x; ++col)
      cout << m[row * size.x + col] << " ";
  }
}

void Test(const dim2& size, const dim2& blockSize) {
  vector<Float> a(size.x * size.y, Float(2));
  vector<Float> b(size.x * size.y, Float(3));
  vector<Float> c(size.x * size.y, Float(0));
  vector<Float> block(blockSize.x * blockSize.y);
  MatMul(a.data(), b.data(), c.data(), block.data(), blockSize, size, size);
  Print(size, c.data());
}

int main(int argc, char** argv) { 
#if 0
  if(argc != 4) {
    cerr << "usage: " << argv[0] << " <num rows> <num columns> <block size>"
         << endl;
    exit(EXIT_FAILURE);
  }
  const dim2 size = {stoul(argv[2]), stoul(argv[1])};
  const dim2 blockSize = {stoul(argv[3]), stoul(argv[3])};
#endif
  const dim2 size = {100, 100};
  const dim2 blockSize = {10, 10};
  Test(size, blockSize);
  return 0;
}
