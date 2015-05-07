#[macro_use]
extern crate calc;
use calc::*;
use std::f64;


struct Square;

impl UnivariateFn for Square {
	#[inline]
	fn eval(&self, x: f64) -> f64 { x*x }
}

fn square_integral(a: f64, b: f64) -> f64 {
	(b*b*b/3f64) - (a*a*a/3f64)
}

fn square_diff(x: f64) -> f64 {
	2f64*x
}

fn isapprox(a: f64, b: f64) -> bool {
	(a-b).abs() < 1e-8f64
}

fn poly2(x: f64, a: f64, b: f64) -> f64 {
	x.mul_add(b, a)
}

fn poly3(x: f64, a: f64, b: f64, c: f64) -> f64 {
	// a + x*(b + x*(c))
	x.mul_add(x.mul_add(c, b), a)
}

fn poly4(x: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
	// a + x*(b + x*(c + x*(d)))
	x.mul_add(x.mul_add(x.mul_add(d, c), b), a)
}

macro_rules! test_square {
	($a:expr, $b:expr) => {
		{
			want!(quad_slow(&Square, $a, $b), square_integral($a, $b));
			want!(quad_fast(&Square, $a, $b), square_integral($a, $b));
			want!(romberg_integral(&Square, $a, $b, 1e-8f64).unwrap(), square_integral($a, $b));
			want!(diff(&Square, $a).unwrap(), square_diff($a));
			want!(diff(&Square, $b).unwrap(), square_diff($b));
		}
	};
}

macro_rules! want {
	($a:expr, $b:expr) => {
		{
			let av = $a;
			let bv = $b;
			if !isapprox(av, bv) {
				println!("Got {:?}; expected {:?}", av, bv);
				assert!(false);
			}
		}
	};
}

#[test]
fn test_integrals() {
	test_square!(0f64, 2f64);
	test_square!(-1f64, 1f64);
	test_square!(1f64, -1f64);
	test_square!(0f64, 27.90821478f64);
	assert!(diff(&Square, f64::NAN).is_none());
	assert!(diff(&Square, f64::INFINITY).is_none());
}

#[test]
fn test_polys() {
	want!(poly!(1f64; 2f64, 3f64), poly2(1f64, 2f64, 3f64));
	want!(poly!(1f64; 2f64, 3f64, 4f64), poly3(1f64, 2f64, 3f64, 4f64));
	want!(poly!(1f64; 2f64, 3f64, 4f64, 5f64), poly4(1f64, 2f64, 3f64, 4f64, 5f64));
}
