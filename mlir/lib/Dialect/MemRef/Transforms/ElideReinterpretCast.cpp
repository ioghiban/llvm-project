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
#include <optional>

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

/// Describes the unique non-unit dimension of a MemRef shape.
///
/// This helper is only used for shapes that have at most one non-unit
/// dimension. `exists` is false for all-ones shapes. Otherwise, `isOnLeft`
/// indicates whether the non-unit dimension is on the left boundary.
///
/// If `exists` is true and `isOnLeft` is false, the non-unit dimension is on
/// the right boundary. Rank-1 non-unit MemRefs are treated as matching both
/// boundaries and callers that care about the right boundary must account for
/// that from the MemRef type.
struct SingleNonUnitDimInfo {
  bool exists = false;
  bool isOnLeft = false;
};

/// Returns information about a MemRef if it contains at most one non-unit
/// dimension.
///
/// The single non-unit dimension, if present, must be on the left or right
/// boundary. Rank-1 non-unit MemRefs are treated as being on both boundaries.
static std::optional<SingleNonUnitDimInfo>
getSingleNonUnitDimInfo(MemRefType type) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t nonUnitCount =
      llvm::count_if(shape, [](int64_t dim) { return dim != 1; });
  // Return default values if missing nonUnitDim
  if (nonUnitCount == 0)
    return SingleNonUnitDimInfo{};
  // Return no info if MemRef breaks nonUnitDim requirements (more nonUnitDims)
  if (nonUnitCount > 1)
    return std::nullopt;

  bool isOnLeft = shape.front() != 1;
  // Return no info if MemRef breaks nonUnitDim requirements (nonUnitDim in
  // non-boundary pos)
  if (!isOnLeft && shape.back() == 1)
    return std::nullopt;

  return SingleNonUnitDimInfo{/*exists=*/true, isOnLeft};
}

static bool hasStaticZeroOffset(memref::ReinterpretCastOp rc) {
  ArrayRef<int64_t> offsets = rc.getStaticOffsets();
  // FIXME: Despite what `getStaticOffsets` implies, `reinterpret_cast` takes
  // only a single offset. That should be fixed at the op definition level.
  assert(offsets.size() == 1 && "Expecting single offset");
  return !ShapedType::isDynamic(offsets[0]) && offsets[0] == 0;
}

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static bool isConstantIndexExplicitlyOutOfBounds(Value idx,
                                                 int64_t upperBound) {
  std::optional<int64_t> idxVal = getConstantIndex(idx);
  return idxVal && (*idxVal < 0 || *idxVal >= upperBound);
}

/// Examples accepted by this shape restriction:
///   memref<999xf32>       <-> memref<1x1x999xf32>
///   memref<1x108xf32>     <-> memref<1x1x1x108xf32>
///   memref<100x1xf32>     <-> memref<100x1x1xf32>
///   memref<1>             <-> memref<1x1x1>
///
/// General reinterpret_casts are intentionally rejected.
static bool isPureRankExpansionOrCollapsingRC(memref::ReinterpretCastOp rc) {
  auto inputTy = cast<MemRefType>(rc.getSource().getType());
  auto outputTy = cast<MemRefType>(rc.getResult().getType());

  // This rewrite assumes "index re-use" and misses "index
  // re-write/adjustment" logic, hence the requirement for the offset to be 0.
  // Thus, storage shift and statically unknown offsets are rejected.
  if (!hasStaticZeroOffset(rc))
    return false;

  // The check assumes the rewrite relies on completely static shape info.
  if (llvm::any_of(rc.getStaticSizes(), ShapedType::isDynamic) ||
      llvm::any_of(rc.getStaticStrides(), ShapedType::isDynamic))
    return false;

  // The check assumes the rewrite supports shapes with at most one non-unit
  // dimension. This excludes underlying multi-dimensional layouts and keeps the
  // rewrite limited to unit-dim insertion/removal `reinterpret_cast`s.
  std::optional<SingleNonUnitDimInfo> inputNonUnitDim =
      getSingleNonUnitDimInfo(inputTy);
  std::optional<SingleNonUnitDimInfo> outputNonUnitDim =
      getSingleNonUnitDimInfo(outputTy);
  // Bail out early if nonUnitDims don't follow rewrite assumptions.
  if (!inputNonUnitDim || !outputNonUnitDim)
    return false;

  // The source and result must either both have a single non-unit dimension
  // or both be all-ones.
  if (inputNonUnitDim->exists != outputNonUnitDim->exists)
    return false;
  if (!inputNonUnitDim->exists)
    return true;

  // The preserved non-unit dimension must have the same size.
  if (inputTy.getDimSize(inputNonUnitDim->isOnLeft ? 0
                                                   : inputTy.getRank() - 1) !=
      outputTy.getDimSize(outputNonUnitDim->isOnLeft ? 0
                                                     : outputTy.getRank() - 1))
    return false;

  // If both sides have rank > 1, the non-unit dimension must be on the same
  // boundary. Rank-1 MemRefs are accepted against either boundary.
  if (inputTy.getRank() != 1 && outputTy.getRank() != 1 &&
      inputNonUnitDim->isOnLeft != outputNonUnitDim->isOnLeft)
    return false;

  return true;
}

/// Checks statically known indices accessed by a load from a pure rank
/// expansion/collapsing to ensure in-bounds only access. Dynamic indices are
/// accepted.
static bool areIndicesInBounds(memref::LoadOp load) {
  auto rc = load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
  auto rcOutputTy = cast<MemRefType>(rc.getResult().getType());

  for (auto [pos, idx] : llvm::enumerate(load.getIndices())) {
    // FIXME: This should be ensured by the memref.load semantics.
    if (isConstantIndexExplicitlyOutOfBounds(idx, rcOutputTy.getDimSize(pos)))
      return false;
  }
  return true;
}

struct RewriteLoadFromReinterpretCast
    : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    auto rc = op.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");
    if (!isPureRankExpansionOrCollapsingRC(rc))
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast is not a pure rank expansion or collapsing of "
              "a single dimension");

    assert(areIndicesInBounds(op) &&
           "load from reinterpret_cast indexes out of bounds!");

    auto rcOutputTy = cast<MemRefType>(rc.getResult().getType());
    auto rcInputTy = cast<MemRefType>(rc.getSource().getType());

    int64_t rcOutputRank = rcOutputTy.getRank();
    int64_t rcInputRank = rcInputTy.getRank();

    SmallVector<Value> idxs(op.getIndices().begin(), op.getIndices().end());
    SmallVector<Value> rcInputIdxs;
    rcInputIdxs.reserve(rcInputRank);

    // The rewrite only supports reinterpret_casts with at most one non-unit
    // dimension, located at the left or right boundary.
    //
    // The higher-rank side tells which side the reinterpret_cast has
    // expanded/collapsed.
    //
    //   expansion: rcOutput has the higher rank
    //   collapsing : rcInput has the higher rank
    //
    // Example:
    //   memref<999>     -> memref<1x1x999>   : extra dims to the left
    //   memref<999x1x1> -> memref<999>       : extra dims to the right
    MemRefType expandedTy =
        rcOutputRank >= rcInputRank ? rcOutputTy : rcInputTy;
    std::optional<SingleNonUnitDimInfo> expandedNonUnitDim =
        getSingleNonUnitDimInfo(expandedTy);
    assert(expandedNonUnitDim && "expected a single boundary non-unit dim");
    bool keepLeadingIndices = expandedNonUnitDim->isOnLeft;

    if (rcOutputRank >= rcInputRank) {
      // Rank expansion:
      //   memref<N>     -> memref<1x1xN> : keep the last rcInputRank indices
      //   memref<N>     -> memref<Nx1x1> : keep the first rcInputRank indices
      //   memref<1>     -> memref<1x1x1> : all indices are zero
      //
      // Any discarded indices are known to be zero from
      // areIndicesInBounds().
      int64_t firstKeptPos =
          keepLeadingIndices ? 0 : rcOutputRank - rcInputRank;
      rcInputIdxs.append(idxs.begin() + firstKeptPos,
                         idxs.begin() + firstKeptPos + rcInputRank);
    } else {
      // Rank collapsing:
      //   memref<1x1xN> -> memref<N>     : reinsert leading zeros
      //   memref<Nx1x1> -> memref<N>     : reinsert trailing zeros
      //   memref<1x1x1> -> memref<1>     : all indices are zero
      //
      // The collapsed-away dimensions are unit dims, so re-adding them with
      // zero indices preserves semantics.
      Value c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      int64_t rankDiff = rcInputRank - rcOutputRank;

      if (keepLeadingIndices) {
        rcInputIdxs.append(idxs.begin(), idxs.end());
        rcInputIdxs.append(rankDiff, c0);
      } else {
        rcInputIdxs.append(rankDiff, c0);
        rcInputIdxs.append(idxs.begin(), idxs.end());
      }
    }

    assert(rcInputIdxs.size() == static_cast<size_t>(rcInputRank) &&
           "Incorrect number of indices!");

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
      return !isPureRankExpansionOrCollapsingRC(rc);
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
  patterns.add<CopyToScalarLoadAndStore, RewriteLoadFromReinterpretCast>(
      patterns.getContext());
}
