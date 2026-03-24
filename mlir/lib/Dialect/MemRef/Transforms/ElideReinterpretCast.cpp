//===-ElideReinterpretCast.cpp - Expansion patterns for MemRef operations-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Repeated.h"
#include <cassert>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELIDEREINTERPRETCASTPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true if `rc` represents a scalar view (all sizes == 1)
/// into a memref that has exactly one non-unit dimension located at
/// either the first or last position (i.e. a "row" or "column").
///
/// Examples that return true:
///
///   // Row-major slice (last dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x1x8xi32> to memref<1x1x1xi32>
///
///   // Column-major slice (first dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<2x1xf32> to memref<1x1xf32>
///
///   // Random strides
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [10, 100]
///     : memref<2x1xf32, strided<[10, 100]>>
///         to memref<1x1xf32>
///
///   // Rank-1 case
///   memref.reinterpret_cast %buf to offset: [%off],
///     sizes: [1], strides: [1]
///     : memref<8xi32> to memref<1xi32>
///
/// Examples that return false:
///
///   // More non-unit dims
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x2x8xi32> to memref<1x1x1xi32>
///
///   // View is not scalar (size != 1)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [2, 1], strides: [1, 1]
///     : memref<1x2xf32> to memref<2x1xf32>
///
///   // Base has non-identity layout
///   %buff = memref.alloc() : memref<1x2xf32, strided<[1, 3]>>
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<1x2xf32, strided<[1, 3]>> to memref<1x1xf32>
static bool isScalarSlice(memref::ReinterpretCastOp rc) {
  auto rcInputTy = dyn_cast<MemRefType>(rc.getSource().getType());
  auto rcOutputTy = dyn_cast<MemRefType>(rc.getType());

  // Reject strided base - logic for computing linear idx is TODO
  if (!rcInputTy.getLayout().isIdentity())
    return false;

  // Reject non-matching ranks
  unsigned srcRank = rcInputTy.getRank();
  if (srcRank != rcOutputTy.getRank())
    return false;

  ArrayRef<int64_t> sizes = rc.getStaticSizes();

  // View must be scalar: memref<1x...x1>
  if (!llvm::all_of(rcOutputTy.getShape(),
                    [](int64_t dim) { return dim == 1; }))
    return false;

  // Sizes must all be statically 1
  if (!llvm::all_of(sizes, [](int64_t size) {
        return !ShapedType::isDynamic(size) && size == 1;
      }))
    return false;

  // Rank-1 special case
  if (srcRank == 1) {
    // Reject non-scalar output
    if (rcOutputTy.getDimSize(0) > 1)
      return false;
  }

  int nonUnitCount =
      std::count_if(rcInputTy.getShape().begin(), rcInputTy.getShape().end(),
                    [](int dim) { return dim != 1; });
  return nonUnitCount == 1;
}

/// Rewrites `memref.copy` of a 1-element MemRef as a scalar load-store pair
///
/// The pattern matches a reinterpret_cast that creates a scalar view
/// (`sizes = [1, ..., 1]`) into a memref with a single non-unit dimension.
/// Since the view contains only one element, the accessed address is
/// determined solely by the base pointer and the offset.
///
/// Two layouts are supported:
///   * row-major slice  (stride pattern [N, ..., 1])
///   * column-major slice (stride pattern [1, ..., N])
///
/// BEFORE (row-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [N, ..., 1]
///       : memref<1x...xNxf32>
///         to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%c0, ..., %off] : memref<1x...xNxf32>
///
/// BEFORE (column-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [1, ..., N]
///       : memref<Nx...x1xf32>
///         to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%off, ..., %c0] : memref<Nx...x1xf32>
struct CopyToScalarLoadAndStore : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value rcOutput = op.getTarget();
    auto rc = rcOutput.getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");

    if (!isScalarSlice(rc))
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar slice");

    Location loc = op.getLoc();

    Value src = op.getSource();
    Value dst = rc.getSource();

    auto dstType = cast<MemRefType>(dst.getType());
    unsigned dstRank = dstType.getRank();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto srcType = cast<MemRefType>(src.getType());
    Repeated<Value> loadIndices(srcType.getRank(), zero);
    auto offsets = rc.getMixedOffsets();
    assert(offsets.size() == 1 && "Expecting single offset");
    OpFoldResult offset = offsets[0];
    Value storeOffset = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
    unsigned offsetDim = dstType.getDimSize(0) == 1 ? dstRank - 1 : 0;
    SmallVector<Value> storeIndices(dstRank, zero);
    storeIndices[offsetDim] = storeOffset;
    // If the only user of `rc` is the current Op (which is about to be erased),
    // we can safely erase it.
    if (rcOutput.hasOneUse())
      rewriter.eraseOp(rc);

    Value val = memref::LoadOp::create(rewriter, loc, src, loadIndices);
    memref::StoreOp::create(rewriter, loc, val, dst, storeIndices);

    rewriter.eraseOp(op);
    return success();
  }
};

static bool isConstZero(Value v) { return matchPattern(v, m_Zero()); }

static bool isPureRankReshape(memref::ReinterpretCastOp rc, memref::LoadOp op) {
  auto inputTy = cast<MemRefType>(rc.getSource().getType());
  auto outputTy = cast<MemRefType>(rc.getResult().getType());

  // This fold only handles reinterpret_casts that behave like pure rank
  // reshapes of a single logical dimension:
  //
  //   - all metadata is static
  //   - offset is 0
  //   - source/result each have at most one non-unit dim
  //   - if a non-unit dim exists, it is at the left or right boundary
  //
  // Examples accepted by this shape restriction:
  //   memref<999xf32>       <-> memref<1x1x999xf32>
  //   memref<1x108xf32>     <-> memref<1x1x1x108xf32>
  //   memref<100x1xf32>     <-> memref<100x1x1xf32>
  //
  // General reinterpret_casts are intentionally rejected.

  auto offsets = rc.getStaticOffsets();
  assert(offsets.size() == 1 && "Expecting single offset");

  // The rewrite drops the reinterpret_cast and remaps indices directly to the
  // source memref. That is only correct if there is no storage shift.
  if (ShapedType::isDynamic(offsets[0]) || offsets[0] != 0)
    return false;

  auto sizes = rc.getStaticSizes();
  auto strides = rc.getStaticStrides();

  // Require fully static metadata. The fold relies on knowing exactly which
  // dimensions are unit dimensions and which indices may be ignored.
  if (llvm::any_of(sizes, ShapedType::isDynamic))
    return false;
  if (llvm::any_of(strides, ShapedType::isDynamic))
    return false;

  // Count non-unit dims and remember their positions.
  //
  // The rewrite supports shapes with at most one non-unit dimension.
  // This excludes underlying multi-dimensional layouts and keeps the
  // fold limited to unit-dim insertion/removal reshapes.
  unsigned inputRank = inputTy.getRank();
  int inputNonUnitCount = 0;
  int64_t inputNonUnitSize = 1;
  unsigned inputNonUnitPos = 0;
  for (unsigned i = 0; i < inputRank; ++i) {
    if (inputTy.getDimSize(i) != 1) {
      ++inputNonUnitCount;
      inputNonUnitPos = i;
      inputNonUnitSize = inputTy.getDimSize(i);
    }
  }

  unsigned outputRank = outputTy.getRank();
  int outputNonUnitCount = 0;
  int64_t outputNonUnitSize = 1;
  unsigned outputNonUnitPos = 0;
  for (unsigned i = 0; i < outputRank; ++i) {
    if (outputTy.getDimSize(i) != 1) {
      ++outputNonUnitCount;
      outputNonUnitPos = i;
      outputNonUnitSize = outputTy.getDimSize(i);
    }
  }

  // Reject reshapes with > 1 non-unit-dimension.
  //
  // The source and result must have the same number of non-unit dimensions:
  // either both are all-ones, or both have exactly one non-unit dimension.
  if (inputNonUnitCount > 1 || outputNonUnitCount > 1 ||
      inputNonUnitCount != outputNonUnitCount)
    return false;

  // If there is a non-unit dimension, it must live at the same boundary
  // (first or last dimension) on both input and output memrefs.
  // The rewrite logic for preserving the load index is exclusive to these
  // cases.
  if (inputNonUnitCount == 1) {
    auto isBoundary = [](unsigned pos, unsigned rank) {
      return pos == 0 || pos == rank - 1;
    };
    if (!isBoundary(inputNonUnitPos, inputRank) ||
        !isBoundary(outputNonUnitPos, outputRank))
      return false;
  }

  // Size of non-unit dimension must be the same
  if (inputNonUnitCount == 1 && outputNonUnitCount == 1 &&
      inputNonUnitSize != outputNonUnitSize)
    return false;

  SmallVector<Value> idxs(op.getIndices().begin(), op.getIndices().end());
  SmallVector<unsigned> nonZeroIdxPositions;
  nonZeroIdxPositions.reserve(idxs.size());

  // Record non-zero indices.
  //
  // During rank expansion, the rewrite drops the extra unit-dimension indices.
  // That is only semantics-preserving if every dropped index is zero.
  for (auto [pos, idx] : llvm::enumerate(idxs)) {
    if (!isConstZero(idx))
      nonZeroIdxPositions.push_back(pos);
  }

  // Position of the unique non-unit dim in the output, if present:
  //   - 0            for shapes like [N, 1, 1]
  //   - outputRank-1 for shapes like [1, 1, N]
  //
  // For the all-ones case, treat it like the "non-unit on the right" case.
  unsigned nonUnitDimPos =
      (outputNonUnitCount == 1 && outputTy.getDimSize(0) != 1) ? 0
                                                               : outputRank - 1;

  if (outputRank >= inputRank) {
    // Rank expansion case.
    //
    // The rewrite keeps only inputRank indices. Any non-zero index in an
    // expanded unit dimension that would be discarded makes the fold invalid.
    if (nonUnitDimPos == 0) {
      // Expansion on the right: keep the leftmost inputRank indices.
      // Therefore any non-zero index in the suffix would be lost.
      for (unsigned pos : nonZeroIdxPositions) {
        if (pos >= inputRank)
          return false;
      }
    } else {
      // Expansion on the left: keep the rightmost inputRank indices.
      // Therefore any non-zero index in the prefix would be lost.
      unsigned firstValidPos = outputRank - inputRank;
      for (unsigned pos : nonZeroIdxPositions) {
        if (pos < firstValidPos)
          return false;
      }
    }
  }

  return true;
}

struct FoldReinterpretCastLoad : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return failure();

    // This fold is only correct for the narrow "pure rank reshape of a single
    // logical dimension" cases accepted by isPureRankReshape().
    if (!isPureRankReshape(rc, op))
      return failure();

    auto rcOutputTy = cast<MemRefType>(rc.getResult().getType());
    auto rcInputTy = cast<MemRefType>(rc.getSource().getType());

    int64_t rcOutputRank = rcOutputTy.getRank();
    int64_t rcInputRank = rcInputTy.getRank();

    SmallVector<Value> idxs(op.getIndices().begin(), op.getIndices().end());
    SmallVector<Value> rcInputIdxs;

    // The fold only supports reshapes with at most one non-unit dimension,
    // located at the left or right boundary.
    //
    // The higher-rank side tells which side the reshape has expanded/collapsed.
    //
    //   expansion: rcOutput has the higher rank
    //   collapse : rcInput has the higher rank
    //
    // Example:
    //   memref<999>     -> memref<1x1x999>   : extra dims to the left
    //   memref<999x1x1> -> memref<999>       : extra dims to the right
    MemRefType expandedTy =
        rcOutputRank >= rcInputRank ? rcOutputTy : rcInputTy;
    bool nonUnitOnLeft = expandedTy.getDimSize(0) != 1;

    if (rcOutputRank >= rcInputRank) {
      // Rank expansion:
      //   memref<N>   -> memref<1x1xN>   : keep the last rcInputRank indices
      //   memref<N>   -> memref<Nx1x1>   : keep the first rcInputRank indices
      //
      // Any discarded indices are known to be zero from isPureRankReshape().
      if (nonUnitOnLeft) {
        for (int64_t dim = 0; dim < rcInputRank; ++dim)
          rcInputIdxs.push_back(idxs[dim]);
      } else {
        for (int64_t dim = 0; dim < rcInputRank; ++dim)
          rcInputIdxs.push_back(idxs[rcOutputRank - rcInputRank + dim]);
      }
    } else {
      // Rank collapse:
      //   memref<1x1xN> -> memref<N>      : reinsert leading zeros
      //   memref<Nx1x1> -> memref<N>      : reinsert trailing zeros
      //
      // The collapsed-away dimensions are unit dims, so readding them with
      // zero indices preserves semantics.
      Value c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      int64_t rankDiff = rcInputRank - rcOutputRank;

      if (nonUnitOnLeft) {
        rcInputIdxs.append(idxs.begin(), idxs.end());
        rcInputIdxs.append(rankDiff, c0);
      } else {
        rcInputIdxs.append(rankDiff, c0);
        rcInputIdxs.append(idxs.begin(), idxs.end());
      }
    }

    // Sanity check: rewritten load must index the source memref with exactly
    // as many indices as the rank.
    if ((int64_t)rcInputIdxs.size() != rcInputRank)
      return failure();

    auto rcInput = rc.getSource();
    // If the only user of rc is the current Op (which is about to be erased),
    // we can safely erase it.
    if (rc.getResult().hasOneUse())
      rewriter.eraseOp(rc);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, rcInput, rcInputIdxs);

    // Do not erase the reinterpret_cast here. After the load is rewritten it
    // may become dead, and canonical DCE can remove it.
    return success();
  }
};

struct ElideReinterpretCastPass
    : public memref::impl::ElideReinterpretCastPassBase<
          ElideReinterpretCastPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    memref::populateElideReinterpretCastPatterns(patterns);
    ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp op) {
      auto rc = op.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !isScalarSlice(rc);
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp op) {
      auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !isPureRankReshape(rc, op);
    });
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateElideReinterpretCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CopyToScalarLoadAndStore, FoldReinterpretCastLoad>(
      patterns.getContext());
}
