(module
  ;; Simple recursive Fibonacci function
  (func $fib (export "fib") (param $n i32) (result i32)
    (if (result i32) (i32.le_s (local.get $n) (i32.const 1))
      (then (local.get $n))
      (else
        (i32.add
          (call $fib (i32.sub (local.get $n) (i32.const 1)))
          (call $fib (i32.sub (local.get $n) (i32.const 2)))
        )
      )
    )
  )

  ;; Iterative Fibonacci for comparison
  (func $fib_iter (export "fib_iter") (param $n i32) (result i32)
    (local $a i32) (local $b i32) (local $i i32) (local $temp i32)
    (local.set $a (i32.const 0))
    (local.set $b (i32.const 1))
    (local.set $i (i32.const 0))
    (if (i32.le_s (local.get $n) (i32.const 0))
      (then (return (i32.const 0)))
    )
    (if (i32.eq (local.get $n) (i32.const 1))
      (then (return (i32.const 1)))
    )
    (block $break
      (loop $continue
        (br_if $break (i32.ge_s (local.get $i) (i32.sub (local.get $n) (i32.const 1))))
        (local.set $temp (i32.add (local.get $a) (local.get $b)))
        (local.set $a (local.get $b))
        (local.set $b (local.get $temp))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )
    (local.get $b)
  )
)
