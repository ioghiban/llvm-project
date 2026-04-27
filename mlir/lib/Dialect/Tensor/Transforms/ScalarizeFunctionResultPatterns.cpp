//===- ScalarizeFunctionResultPatterns.cpp - Scalarize tensor returns -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

struct ScalarizeSingleElementTensorReturnPattern
    : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (funcOp.isDeclaration())
      return rewriter.notifyMatchFailure(funcOp, "function has no body");

    FunctionType functionType = funcOp.getFunctionType();
    if (functionType.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          funcOp, "function does not return exactly one value");

    auto tensorType = dyn_cast<RankedTensorType>(functionType.getResult(0));
    if (!tensorType)
      return rewriter.notifyMatchFailure(
          funcOp, "function result is not a ranked tensor");
    if (!tensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          funcOp, "function result tensor does not have a static shape");
    if (tensorType.getNumElements() != 1)
      return rewriter.notifyMatchFailure(
          funcOp, "function result tensor does not have exactly one element");

    Operation *symbolTable =
        SymbolTable::getNearestSymbolTable(funcOp->getParentOp());
    if (symbolTable && !SymbolTable::symbolKnownUseEmpty(funcOp, symbolTable))
      return rewriter.notifyMatchFailure(funcOp, "function has symbol users");

    SmallVector<func::ReturnOp> returnOps;
    for (Block &block : funcOp.getBody()) {
      auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
      if (!returnOp)
        return rewriter.notifyMatchFailure(
            funcOp, "function has a non-func.return terminator");
      if (returnOp.getNumOperands() != 1)
        return rewriter.notifyMatchFailure(
            returnOp, "return does not have exactly one operand");
      assert(returnOp.getOperand(0).getType() == tensorType &&
             "return operand type must match function result type");
      returnOps.push_back(returnOp);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    Value zeroIndex =
        arith::ConstantIndexOp::create(rewriter, funcOp.getLoc(), 0);

    SmallVector<Value> zeroIndices(tensorType.getRank(), zeroIndex);

    Type scalarType = tensorType.getElementType();
    for (func::ReturnOp returnOp : returnOps) {
      rewriter.setInsertionPoint(returnOp);
      Value extracted = tensor::ExtractOp::create(
          rewriter, returnOp.getLoc(), returnOp.getOperand(0), zeroIndices);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, extracted);
    }

    SmallVector<Type> newResults{scalarType};
    FunctionType newFunctionType = FunctionType::get(
        funcOp.getContext(), functionType.getInputs(), newResults);
    rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newFunctionType); });
    return success();
  }
};

} // namespace

void mlir::tensor::populateScalarizeSingleElementTensorReturnPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ScalarizeSingleElementTensorReturnPattern>(
      patterns.getContext());
}
