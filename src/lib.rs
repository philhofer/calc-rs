use std::f64;
const K: usize = 10;

/// UnivariateFn is the trait that
/// is implemented by univariate functions.
pub trait UnivariateFn {
	fn eval(&self, f64) -> f64;
}

pub fn integral<T: UnivariateFn>(f: T, a: f64, b: f64) -> f64 {
	if a.is_nan() || b.is_nan() {
		return f64::NAN;
	}
	if a == b {
		return 0f64;
	}
	if a.is_infinite() || b.is_infinite() {
		let (x, y) = inf_integral(a, b);
		let bt = BoundsTranform{inner: &f};
		return raw_integral(&bt, x, y);
	}
	if b < a {
		-1f64 * raw_integral(f, b, a)
	} else {
		raw_integral(f, a, b)
	}
}

// integral w/o NAN/INFINITY checks, etc.
fn raw_integral<T: UnivariateFn>(f: T, a: f64, b: f64) -> f64 {
	let mut k0 = [0f64; K];
	let (beg, end) = (f.eval(a), f.eval(b));
	for i in 0..K {
		let n = 1<<i;
		k0[i] = match n {
			0 => a,
			1 => (b - a) * (end - beg) * 0.5f64,

			// perform trapezoidal integration in 'n' steps
			_ => {
				let h = (b - a) / (n as f64);
				let mut middle = 0f64;
				let mut pos = a + h;
				for _ in 1..n {
					middle += f.eval(pos);
					pos += h;
				}
				let last = f.eval(pos);
				// before fma magic:
				// (h/2)*(first + 2.0*middle + last)
				h*(beg + middle.mul_add(2f64, last))*0.5f64
			}
		};
	}
	println!("{:?}", k0);
	for k in 1..K {
		let lk = 1<<(2*k);
		for i in 0..K-k {
			k0[i] = ((lk as f64)*k0[i+1]-k0[i])/((lk-1) as f64);
		}
		if isapprox(k0[K-k-1], k0[K-k]) {
			return k0[K-k-1];
		}
	}
	k0[1]
}

// shim for transforming bounds
struct BoundsTranform<'a, T: 'a> {
	inner: &'a T,
}

impl<'a, T: UnivariateFn> UnivariateFn for &'a BoundsTranform<'a, T> {
	fn eval(&self, z: f64) -> f64 {
		let zsm = z.mul_add(z, -1f64);
		let zsq = zsm + 1f64;
		self.inner.eval( -z / ((z-1f64)*(z+1f64)) ) * (zsq + 1f64) / (zsm * zsm)
	}
}

// shim for simple univariate functions
impl UnivariateFn for fn(f64)->f64 {
	#[inline]
	fn eval(&self, x: f64) -> f64 { (*self)(x) }
}

fn inf_integral(low: f64, high: f64) -> (f64, f64) {
	(match low {
		0f64 => 0f64,
		f64::NEG_INFINITY => -1f64 + 1e-15f64,
		_ => ((4f64*low*low+1f64).sqrt()-1f64) / (2f64*low),
	},
	match high {
		0f64 => 0f64,
		f64::INFINITY => 1f64 - 1e-15f64,
		_ => ((4f64*high*high+1f64).sqrt()-1f64) / (2f64*low),
	})
}

#[inline]
pub fn isapprox(a: f64, b: f64) -> bool {
	(a - b).abs() < 1e-10f64
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::f64;

	struct Square;

	impl UnivariateFn for Square {
		fn eval(&self, x: f64) -> f64 { x*x }
	}

	struct Exp {deg: f64}

	impl UnivariateFn for Exp {
		fn eval(&self, x: f64) -> f64 {
			(self.deg*x).exp()
		}
	 }

	fn square_integral(a: f64, b: f64) -> f64 {
		(b*b*b/3f64) - (a*a*a/3f64)
	}

	macro_rules! test_square {
		($a:expr, $b:expr) => {
			{
				want!(integral(Square, $a, $b), square_integral($a, $b));
			}
		};
	}

	macro_rules! want {
		($a:expr, $b:expr) => {
			{
				let av = $a;
				let bv = $b;
				if !isapprox(av, bv) {
					println!("Got {}; expected {}", av, bv);
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
		want!(integral(Exp{deg: -1f64}, 0f64, f64::INFINITY), 1f64);
	}
}


