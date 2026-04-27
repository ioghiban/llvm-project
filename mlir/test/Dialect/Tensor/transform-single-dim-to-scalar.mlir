// RUN: mlir-opt -test-tensor-transform-patterns=test-scalarize-single-element-tensor-return %s | FileCheck %s

// Inserted ExtractOp gets constant folded for rank-0 tensors
// i.e. no accessed indices
func.func @rank0() -> tensor<i64> {
  %0 = arith.constant dense<-1> : tensor<i64>
  return %0 : tensor<i64>
}
// CHECK-LABEL: func.func @rank0
//  CHECK-SAME:     -> i64
//  CHECK-NEXT:   %[[CST:.*]] = arith.constant -1 : i64
//  CHECK-NEXT:   return %[[CST]] : i64

func.func @rank1(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @rank1
// CHECK-SAME:      %[[SRC:.*]]: tensor<1xi64>) -> i64 {
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]]] : tensor<1xi64>
//      CHECK:   return %[[EXT]] : i64

func.func @rank2_single_element(%arg0: tensor<1x1xi64>) -> tensor<1x1xi64> {
  return %arg0 : tensor<1x1xi64>
}
// CHECK-LABEL: func.func @rank2_single_element
//  CHECK-SAME:     %[[SRC:.*]]: tensor<1x1xi64>) -> i64
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[SRC]][%[[C0]], %[[C0]]] : tensor<1x1xi64>
//       CHECK:   return %[[EXT]] : i64

func.func @caller(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  %0 = func.call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @caller
//  CHECK-SAME:     -> i64
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[CALL:.*]] = call @callee(%arg0) : (tensor<1xi64>) -> tensor<1xi64>
//       CHECK:   %[[EXT:.*]] = tensor.extract %[[CALL]][%[[C0]]] : tensor<1xi64>
//  CHECK-NEXT:   return %[[EXT]] : i64

//===----------------------------------------------------------------------===//
// Negative tests (must NOT rewrite)
//===----------------------------------------------------------------------===//

func.func @multiple_elements(%arg0: tensor<2xi64>) -> tensor<2xi64> {
  return %arg0 : tensor<2xi64>
}
// CHECK-LABEL: func.func @multiple_elements
//  CHECK-SAME:     -> tensor<2xi64>
//   CHECK-NOT:   tensor.extract
//  CHECK-NEXT:   return %arg0 : tensor<2xi64>

/// If no @caller, then rewrite applied to @callee
func.func @callee(%arg0: tensor<1xi64>) -> tensor<1xi64> {
  return %arg0 : tensor<1xi64>
}
// CHECK-LABEL: func.func @callee
//  CHECK-SAME:     -> tensor<1xi64>
//   CHECK-NOT:   tensor.extract
//  CHECK-NEXT:   return %arg0 : tensor<1xi64>