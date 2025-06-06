; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5

; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

define i32 @bitcast_failure(ptr %0, <1 x i16> %1) {
; CHECK-LABEL: bitcast_failure:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov x8, x0
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    // kill: def $d0 killed $d0 def $q0
; CHECK-NEXT:    str h0, [x8]
; CHECK-NEXT:    ret
  %3 = bitcast <1 x i16> %1 to <1 x half>
  %4 = extractelement <1 x half> %3, i64 0
  store half %4, ptr %0, align 2
  ret i32 0
}
