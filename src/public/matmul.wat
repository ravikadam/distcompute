(module
  (memory (export "mem") 16)
  ;; C[M,N] = A[M,K] @ B[K,N], contiguous row-major f32. Pointers are byte offsets.
  (func (export "matmul")
    (param $a i32) (param $b i32) (param $c i32) (param $M i32) (param $K i32) (param $N i32)
    (local $i i32) (local $j i32) (local $k i32) (local $jN i32)
    (local $acc v128) (local $av v128) (local $cacc f32) (local $bptr i32) (local $aik i32)
    (local.set $jN (i32.and (local.get $N) (i32.const 0xFFFFFFFC)))
    (local.set $i (i32.const 0))
    (block $iend (loop $iloop
      (br_if $iend (i32.ge_s (local.get $i) (local.get $M)))
      ;; vectorized columns (4 at a time)
      (local.set $j (i32.const 0))
      (block $jend (loop $jloop
        (br_if $jend (i32.ge_s (local.get $j) (local.get $jN)))
        (local.set $acc (v128.const i32x4 0 0 0 0))
        (local.set $k (i32.const 0))
        (block $kend (loop $kloop
          (br_if $kend (i32.ge_s (local.get $k) (local.get $K)))
          (local.set $av (f32x4.splat
            (f32.load (i32.add (local.get $a)
              (i32.shl (i32.add (i32.mul (local.get $i) (local.get $K)) (local.get $k)) (i32.const 2))))))
          (local.set $bptr (i32.add (local.get $b)
            (i32.shl (i32.add (i32.mul (local.get $k) (local.get $N)) (local.get $j)) (i32.const 2))))
          (local.set $acc (f32x4.add (local.get $acc)
            (f32x4.mul (local.get $av) (v128.load (local.get $bptr)))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br $kloop)))
        (v128.store
          (i32.add (local.get $c) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $N)) (local.get $j)) (i32.const 2)))
          (local.get $acc))
        (local.set $j (i32.add (local.get $j) (i32.const 4)))
        (br $jloop)))
      ;; scalar tail columns [jN, N)
      (local.set $j (local.get $jN))
      (block $tend (loop $tloop
        (br_if $tend (i32.ge_s (local.get $j) (local.get $N)))
        (local.set $cacc (f32.const 0))
        (local.set $k (i32.const 0))
        (block $tkend (loop $tkloop
          (br_if $tkend (i32.ge_s (local.get $k) (local.get $K)))
          (local.set $cacc (f32.add (local.get $cacc)
            (f32.mul
              (f32.load (i32.add (local.get $a) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $K)) (local.get $k)) (i32.const 2))))
              (f32.load (i32.add (local.get $b) (i32.shl (i32.add (i32.mul (local.get $k) (local.get $N)) (local.get $j)) (i32.const 2)))))))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br $tkloop)))
        (f32.store (i32.add (local.get $c) (i32.shl (i32.add (i32.mul (local.get $i) (local.get $N)) (local.get $j)) (i32.const 2))) (local.get $cacc))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $tloop)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $iloop)))))
