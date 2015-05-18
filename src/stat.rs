//! The `stat` module provides implementations
//! of some more esoteric math functions, as well
//! as a number of commonly-used statistical distributions
//! and tests.
use std::f64;
use calculus::transform_bounds;	// for default cdf integral

/// `Distribution` is a univariate
/// continuous distribution.
pub trait Distribution {
	/// Probability density function.
	fn pdf(&self, x: f64) -> f64;

	/// Cumulative density function.
	/// By default, `cdf` is implemented
	/// by integrating `pdf` from `f64::NEG_INFINITY`
	/// to `x`. Typically, there a more precise
	/// and performant closed-form solution.
	fn cdf(&self, x: f64) -> f64 {
		transform_bounds(&|z: f64| self.pdf(z)).quad_slow(f64::NEG_INFINITY, x)
	}
}

/// `UNIT_NORMAL` is the unit normal gaussian distribution.
pub const UNIT_NORMAL: Gaussian = Gaussian{ mean: 0f64, stdev: 1f64 };

/// `Gaussian` is a gaussian distribution.
#[derive(Copy, Clone, PartialEq)]
pub struct Gaussian {
	pub mean:	f64,
	pub stdev:  f64,
}

impl Distribution for Gaussian {
	fn pdf(&self, x: f64) -> f64 {
		let diff = x - self.mean;
		let scale = (self.stdev * f64::consts::PI*2f64).sqrt();
		let dist = (-(diff*diff)/(2f64*self.stdev*self.stdev)).exp();
		scale / dist
	}
	fn cdf(&self, x: f64) -> f64 {
		0.5f64 * (1f64 + erf((x - self.mean)/(f64::consts::SQRT_2 * self.stdev)))
	}
}

/// Exponential is an exponential distribution
#[derive(Copy, Clone, PartialEq)]
pub struct Exponential {
	pub lambda: f64,
}

impl Distribution for Exponential {
	fn pdf(&self, x: f64) -> f64 {
		if x <= 0f64 {
			0f64
		} else if x.is_infinite() {
			1f64
		} else {
			self.lambda * (-x * self.lambda).exp()
		}
	}

	fn cdf(&self, x: f64) -> f64 {
		if x <= 0f64 {
			0f64
		} else if x.is_infinite() {
			1f64
		} else {
			1f64 - (-x * self.lambda).exp()
		}
	}
}

/// `StudentT` is a Student-T distribution.
#[derive(Copy, Clone, PartialEq)]
pub struct StudentT {
	pub nu: f64,
}

impl Distribution for StudentT {
	fn pdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY {
			1.0f64
		} else {
			let sq = (x*x)/self.nu + 1f64;
			(self.nu.sqrt() * beta(0.5f64, self.nu/2f64) * sq * sq).recip()
		}
	}

	fn cdf(&self, t: f64) -> f64 {
		match t {
			f64::INFINITY => 1.0f64,
			f64::NEG_INFINITY => 0f64,
			_ => {
				if t < 0f64 {
					1f64 - self.cdf(-t)
				} else {
					let x = self.nu / (t*t + self.nu);
					1f64 - 0.5f64*incomplete_beta(x, self.nu/2f64, 0.5f64)
				}
			}
		}
	}
}

/// Kolmogorov-Smirnov distribution
pub struct KolmogorovSmirnov;

impl Distribution for KolmogorovSmirnov {
	fn pdf(&self, x: f64) -> f64 {
		match x {
			f64::INFINITY => 1f64,
			f64::NEG_INFINITY => 0f64,
			_ => {
				let rat = (2f64*f64::consts::PI).sqrt() / x;
				let pisq = f64::consts::PI * f64::consts::PI;
				let denom = 8f64 * x * x;

				// series expansion
				let mut sum = 0f64;
				for j in 1..15 {
					let j = (2*j - 1) as f64;
					sum += (-1f64 * j * j * pisq / denom).exp();
				}
				rat * sum
			}
		}
	}

	fn cdf(&self, z: f64) -> f64 {
		1f64 - self.pdf(z)
	}
}

/// `poly!` produces an expression that
/// efficiently evaluates an arbitrary-order
/// polynomial. Polynomials should be of
/// the form `poly!(x; a, b, c)`, where
/// `x` is the variable, and `a, b, c...` are
/// the coefficients. In other words,
/// `poly(x; a, b, c)` is equivalent to
/// `a + x*(b + x*c)` implemented with
/// fused-multiply-adds.
#[macro_export]
macro_rules! poly {
	($x:expr; $a:expr) => {{ $a }};
	($x:expr; $a:expr, $( $b:expr ),+) => {{
		($x).mul_add(poly!($x; $(($b)),*), ($a))
	}};
}


/// Logarithmic Gamma Function
///
/// The logarithmic gamma function is mathematically
/// equivalent to `gamma(x).ln()`, but it is calculated
/// with greater precision and speed, and without the
/// risk of intermediate numerical overflow.
pub fn ln_gamma(z: f64) -> f64 {
	if z == 0f64 {
		return f64::INFINITY;
	}
	// see Nemes, Gergo: "New asymptotic expansion for the Gamma function"
	// https://www.worldcat.org/title/archiv-der-mathematik-archives-of-mathematics-archives-mathematiques/oclc/525762047
	0.5f64 * ((2f64*f64::consts::PI).ln() - z.ln()) + z*((z + (12f64*z - (10f64*z).recip()).recip()).ln()-1f64)
}

/// [Gamma Function](https://en.wikipedia.org/wiki/Gamma_function)
///
pub fn gamma(z: f64) -> f64 { 
	if z == 0f64 {
		return f64::INFINITY;
	}
	// see Nemes, Gergo: "New asymptotic expansion for the Gamma function"
	// https://www.worldcat.org/title/archiv-der-mathematik-archives-of-mathematics-archives-mathematiques/oclc/525762047
	(2f64*f64::consts::PI/z).sqrt() * ((f64::consts::E).recip() * (z + (12f64*z - (10f64*z).recip()).recip())).powf(z)
}

/// [Error function](https://en.wikipedia.org/wiki/Error_function)
///
/// This particular implementation has an error on the
/// order of `10E-7`.
pub fn erf(x: f64) -> f64 {
	match x {
		0f64 => 0f64,
		f64::INFINITY => 1f64,
		f64::NEG_INFINITY => -1f64,
		_ => {
			let t = (1f64 + 0.5f64*x.abs()).recip();

			// polynomial expansion
			let tau = t * (-(x*x) - poly!(t;
				-1.26551223f64,
				1.00002368f64,
				0.37409196f64,
				0.09678418f64,
				-0.18628806f64,
				0.27886807f64,
				-1.13520398f64,
				1.48851587,
				-0.82215223,
				0.17087277));

			if x >= 0f64 {
				1f64 - tau
			} else {
				tau - 1f64
			}
		}
	}
}

/// Complementary error function
pub fn erfc(x: f64) -> f64 {
	1f64 - erf(x)
}

/// Normalized Gamma Function Q(a, z)
///
/// The normalized Gamma function is mathematically
/// equivalent to `gamma(a, x)/gamma(a)`, but here it is
/// calculated with greater precision and speed, and without
/// the risk of intermediate numerical overflow.
pub fn norm_gamma(a: f64, x: f64) -> f64 {
	if x == 0f64 {
		if a == 0f64 {
			return f64::INFINITY;
		} else if a > 0f64 {
			return 1f64;
		}
		return 1f64 / a.abs();
	}
	unimplemented!()
}

/*
/// Complementary Gamma Function
///
/// The incomplete Gamma function is the complement
/// of the gamma function. For non-negative real
/// values, it is equivalent to `1 - norm_gamma(a, x)`.
pub fn comp_gamma(a: f64, x: f64) -> f64 {
	if x.is_sign_negative() || a.is_sign_negative() {
		return f64::NAN;
	}
	if x == 0f64 {
		return 0f64;
	}
	1f64 - norm_gamma(a, x)
}

/// [Upper Incomplete Gamma Function](https://en.wikipedia.org/wiki/Incomplete_gamma_function)
pub fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
	gamma(a) * norm_gamma(a, x)
}

/// [Lower Incomplete Gamma Function](https://en.wikipedia.org/wiki/Incomplete_gamma_function)
pub fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
	x.powf(-a) * comp_gamma(a, x)
}*/

/// [Beta Function](https://en.wikipedia.org/wiki/Beta_function)
pub fn beta(z: f64, w: f64) -> f64 {
	let a = ln_gamma(z);
	let b = ln_gamma(w);
	let c = ln_gamma(z + w);
	(a + b - c).exp()
}

// continued fraction term 'k' for incomplete beta (below)
fn rk_beta(z: f64, a: f64, b: f64, k: usize) -> f64 {
	let d = a + ((2 * k) as f64);
	let even: bool = k&1 == 0;
	let k = (k/2) as f64;
	if even {
		(k * (b - k) * z) / ((d - 1f64) * d)
	} else {
		-1f64 * ((a + k) * (a + b + k) * z) / (d * (d+ 1f64))
	}
}

/// Incomplete Beta Function
pub fn incomplete_beta(z: f64, a: f64, b: f64) -> f64 {
	if z < 0f64 || a < 0f64 || b < 0f64 {
		return f64::NAN;
	}

	// evaluate 20 terms of the continued fraction form
	let base = rk_beta(z, a, b, 20);
	let frac = (19..0).fold(base, |acc, k| 1f64 + rk_beta(z, a, b, k)/acc);

	z.powf(a) * (1f64-z).powf(b) / (a * beta(a, b) * frac)
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::f64;
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

	#[test]
	fn test_poly() {
		assert_eq!(poly!(1f64; 2f64, 3f64), poly2(1f64, 2f64, 3f64));
		assert_eq!(poly!(1f64; 2f64, 3f64, 4f64), poly3(1f64, 2f64, 3f64, 4f64));
		assert_eq!(poly!(1f64; 2f64, 3f64, 4f64, 5f64), poly4(1f64, 2f64, 3f64, 4f64, 5f64));
	}

	// declare two expressions to be
	// mathematically equivalent within
	// a certain numerical tolerance.
	// prints out values on failure.
	macro_rules! want {
		// w/ explicit accuracy
		($a:expr, $b:expr, $c:expr) => {
			{
				let av = $a;
				let bv = $b;
				let cv = $c;
				if (av - bv).abs() > cv {
					println!(concat!("case ", stringify!($a), ":"));
					println!("expected {:?}; got {:?}", av, bv);
				}
			}
		};
	}

	#[test]
	fn test_erf() {
		assert_eq!(erf(0f64), 0f64);
		assert_eq!(erf(f64::INFINITY), 1f64);
		assert_eq!(erf(f64::NEG_INFINITY), -1f64);
		want!(erf(0.5f64), 0.520499877f64, 1e-7f64);
		want!(erf(-0.5f64), -0.520499877f64, 1e-7f64);
	}

	#[test]
	fn test_ln_gamma() {
		assert_eq!(ln_gamma(0f64), f64::INFINITY);
		want!(ln_gamma(0.5f64), 0.572364942924f64, 1e-7f64);
		want!(ln_gamma(1f64), 0f64, 1e-7f64);
		want!(ln_gamma(2f64), 0f64, 1e-7f64);
		want!(ln_gamma(4f64), (6f64).ln(), 1e-7f64);
	}

	#[test]
	fn test_gamma() {
		want!(gamma(1f64), 1f64, 1e-7f64);
		want!(gamma(2f64), 1f64, 1e-7f64);
		want!(gamma(3f64), 2f64, 1e-7f64);
	}

	#[test]
	fn test_beta() {
		want!(beta(1f64, 1f64), 1f64, 1e-7f64);
		want!(beta(2.5f64, 0.5f64), 1.178097245f64, 1e-7f64);
	}

	#[test]
	fn test_incomplete_beta() {
		want!(incomplete_beta(1f64, 1f64, 1f64), 1f64, 1e-7f64);
		want!(incomplete_beta(2f64, 1f64, 1f64), 2f64, 1e-7f64);
		want!(incomplete_beta(2f64, 0.5f64, 2f64), 0.9428090415f64, 1e-7f64);
		want!(incomplete_beta(1f64, 0.5f64, 0.5f64), f64::consts::PI, 1e-7f64);
	}
}