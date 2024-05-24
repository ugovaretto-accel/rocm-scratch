/// Block matrix-matrix multiply with non square matrices and non-square blocks
/// @author Ugo Varetto

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

using Float = float;

/// Used for dimensions (x = columns, y = rows) and offsets.
struct dim2 {
    size_t x = 0;
    size_t y = 0;
};

/// Swap x and y.
dim2 Invert(const dim2& d) { return {d.y, d.x}; }

/// Compute dimension of matrix resulting from multiplying two matrices
/// of sizes b1 and b1.
dim2 DimMul(const dim2& b1, const dim2& b2) { return {b1.y, b2.x}; }

/// Print matrix
void Print(const dim2& size, const Float* m) {
    for (size_t row = 0; row != size.y; ++row) {
        for (size_t col = 0; col != size.x; ++col)
            cout << m[row * size.x + col] << " ";
        cout << endl;
    }
}

/// Multiply one sub-matrix from matrix A, one sub-matrix from matrix B
/// and store the result as a sub-matrix of C. Note that A, B, C can be
/// matrices of arbitrary size with no requirement of having rows(C) == rows(A)
/// and columns(C) == columns(B).
///
/// @param a input matrix A
/// @param b input matrix B
/// @param c output matrix C
/// @param blockDimA dimensions of A sub-matrix
/// @param blockDimB dimensions of B sub-batrix
/// @param dimensions of matrix A
/// @param dimensions of matrix B
/// @param dimensions of matrix C
/// @param offsetA location of input sub-matrix inside A
/// @param offsetB location of input sub-batrix inside B
/// @param offsetB location of output sub-matrix inside C
void BlockMul(const Float* a, const Float* b, Float* c, dim2 blockDimA,
              dim2 blockDimB, dim2 dimA, dim2 dimB, dim2 dimC, dim2 offsetA,
              dim2 offsetB, dim2 offsetC) {
    const size_t nAcols = dimA.x;
    const size_t nArows = dimA.y;
    const size_t nBcols = dimB.x;
    const size_t nBrows = dimB.y;
    const size_t nABlockRows = blockDimA.y;
    const size_t nABlockCols = blockDimB.x;
    const size_t nBBlockCols = blockDimB.x;
    const size_t nCcols = dimC.x;
    for (size_t row = 0; row != nABlockRows; ++row) {
        for (size_t col = 0; col != nBBlockCols; ++col) {
            Float v = Float(0);
            for (size_t x = 0; x != nABlockCols; ++x) {
                v += a[(row + offsetA.y) * nAcols + offsetA.x + x] *
                     b[(x + offsetB.y) * nBcols + offsetB.x + col];
            }
            c[(row + offsetC.y) * nCcols + offsetC.x + col] = v;
        }
    }
}

/// Add sub-matrix to sub-matrix in place.
///
/// @param a input matrix A
/// @param c output matrix C
/// @param dimA dimensions of A
/// @param dimC dimensions of C
/// @param blockDimA dimensions or A sub-matrix == dimensions of C sub-matrix
/// @param offsetA location of sub-matrix in matrix A
/// @param offsetC location of sub-matrix in matrix C
void InplaceMatAdd(const Float* a, Float* c, dim2 dimA, dim2 dimC,
                   dim2 blockDimA, dim2 offsetA, dim2 offsetC) {
    const size_t nAcols = dimA.x;
    const size_t nArows = dimA.y;
    const size_t nABlockRows = blockDimA.y;
    const size_t nABlockCols = blockDimA.x;
    const size_t nCcols = dimC.x;
    const size_t nCrows = dimC.y;
    for (size_t row = 0; row != nABlockRows; ++row) {
        for (size_t col = 0; col != nABlockCols; ++col) {
            c[(offsetC.y + row) * nCcols + offsetC.x + col] +=
                a[(offsetA.y + row) * nAcols + offsetA.x + col];
        }
    }
}

/// Generic matrix-matrix multiply algorithm, use blockDim = {1, 1}
/// if no blocking required.
///
/// @param a input matrix A
/// @param b input matrix B
/// @param c output matrix C
/// @param b cache to store intermediate multiply results
/// @param blockDim block dimension, does not need to be square,
///                 if not square block size in matrix B is computed
///                 such as C blocks are square
/// @param dimA matrix A dimensions
/// @param dimB matrix B dimensions
void MatMul(const Float* a, const Float* b, Float* c, Float* block,
            dim2 blockDim, dim2 dimA, dim2 dimB) {
    assert(a);
    assert(b);
    assert(c);
    assert(block);
    assert(blockDim.x > 0 && blockDim.y > 0);
    assert(dimA.x > 0 && dimA.y > 0);
    assert(dimB.x > 0 && dimB.y > 0);
    const size_t nAcols = dimA.x / blockDim.x;
    const size_t nArows = dimA.y / blockDim.y;
    const dim2 blockDimB = Invert(blockDim);
    const dim2 blockDimC = DimMul(blockDim, blockDimB);
    const size_t nBcols = dimB.x / blockDimB.x;
    const size_t nBrows = dimB.y / blockDimB.y;
    const size_t nCcols = nBcols;
    const size_t nCrows = nArows;
    const dim2 dimC = {dimB.x, dimA.y};
    const dim2 offsetBlock = {0, 0};
    for (size_t row = 0; row != nArows; ++row) {
        for (size_t col = 0; col != nBcols; ++col) {
            const dim2 offsetC = {col * blockDim.x, row * blockDim.y};
            for (size_t y = 0; y != nAcols; ++y) {
                // C[row][col] = A[row][c] x B[c][col];
                const dim2 offsetA = {y * blockDim.x, row * blockDim.y};
                const dim2 offsetB = {col * blockDim.x, y * blockDim.y};
                BlockMul(a, b, block, blockDim, blockDimB, dimA, dimB,
                         blockDimC, offsetA, offsetB, offsetBlock);
                InplaceMatAdd(block, c, blockDimC, dimC, blockDimC, {0, 0},
                              offsetC);
            }
        }
    }
}

/// Generate matrices, multiply and print.
void Test(const dim2& size, const dim2& blockSize) {
    vector<Float> a(size.x * size.y, Float(1));
    vector<Float> b(size.x * size.y, Float(1));
    vector<Float> c(size.x * size.y, Float(0));
    vector<Float> block(blockSize.x * blockSize.y);
    MatMul(a.data(), b.data(), c.data(), block.data(), blockSize, size, size);
    Print(size, c.data());
}

/// Invoke matrix multiply test
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
    const dim2 size = {50, 50};
    const dim2 blockSize = {5, 5};
    Test(size, blockSize);
    return 0;
}
