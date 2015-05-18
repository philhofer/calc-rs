//! The `calculus` module contains utilities
//! for performing integration and differentiation.
//! 
use std::f64;

/// `UnivariateFn` is a function that 
/// takes one `f64` and returns another.
/// `eval` must always return the same
/// value for a given input.
pub trait UnivariateFn {
	fn eval(&self, x: f64) -> f64;
}

/// Univariate functions/closures of type
/// `fn(f64)->f64` implicitly implement `UnivariateFn`.
impl<T> UnivariateFn for T where T: Fn(f64) -> f64 {
	#[inline]
	fn eval(&self, x: f64) -> f64 { self(x) }
}

// 1st-order central diff approximation
#[inline]
fn fa0<T: UnivariateFn>(f: &T, x: f64, h: f64) -> f64 {
	(f.eval(x+h)-f.eval(x-h))/(2f64*h)
}

// 1st Richardson extrapolation
#[inline]
fn fa1<T: UnivariateFn>(f: &T, x: f64, h: f64) -> f64 {
	(4f64*fa0(f, x, h/2f64)-fa0(f, x, h))/3f64
}

// 2nd Richardson extrapolation
#[inline]
fn fa2<T: UnivariateFn>(f: &T, x: f64, h: f64) -> f64 {
	(16f64*fa1(f, x, h/4f64)-fa1(f, x, h))/15f64
}

/// Calculates the derivative of the input function at the point
/// `x`. Returns `None` if the derivative is not numerically stable,
/// or if `x` is `NAN` or infinite.
///
/// # Example
/// ```
/// use calc::calculus::*;
/// 
/// // the derivative of x^2 at 1 is 2
/// assert!((diff(&|x: f64| x*x, 1f64).unwrap() - 2f64).abs() < 1e-7f64);
/// ```
pub fn diff<T: UnivariateFn>(f: &T, x: f64) -> Option<f64> {
	if x.is_nan() || x.is_infinite() {
		return None;
	}
	const H0: f64 = 0.00316227766016837933199;
	let (a1, a2, a3) = (fa0(f, x, H0), fa1(f, x, H0), fa2(f, x, H0));
	// we've converged if the 2nd- and 3rd-order
	// approximations are closer than the 1st and 2nd, within
	// numerical precision.
	if ((a1-a2).abs() - (a2-a3).abs()) > -1e-8f64 {
		return Some(a3);
	}
	None
}

// 25 points of Gauss-Legendre weights and abscissae
// (to be used symmetrically)
const GAUSS_TAB_25: [(f64, f64); 25] = [
	(0.0621766166553473f64, 0.0310983383271889f64),
	(0.0619360674206832f64, 0.0931747015600861f64),
	(0.0614558995903167f64, 0.1548905899981459f64),
	(0.0607379708417702f64, 0.2160072368760418f64),
	(0.0597850587042655f64, 0.2762881937795320f64),
	(0.0586008498132224f64, 0.3355002454194373f64),
	(0.0571899256477284f64, 0.3934143118975651f64),
	(0.0555577448062125f64, 0.4498063349740388f64),
	(0.0537106218889962f64, 0.5044581449074642f64),
	(0.0516557030695811f64, 0.5571583045146501f64),
	(0.0494009384494663f64, 0.6077029271849502f64),
	(0.0469550513039484f64, 0.6558964656854394f64),
	(0.0443275043388033f64, 0.7015524687068222f64),
	(0.0415284630901477f64, 0.7444943022260685f64),
	(0.0385687566125877f64, 0.7845558329003993f64),
	(0.0354598356151462f64, 0.8215820708593360f64),
	(0.0322137282235780f64, 0.8554297694299461f64),
	(0.0288429935805352f64, 0.8859679795236131f64),
	(0.0253606735700124f64, 0.9130785566557919f64),
	(0.0217802431701248f64, 0.9366566189448780f64),
	(0.0181155607134894f64, 0.9566109552428079f64),
	(0.0143808227614856f64, 0.9728643851066920f64),
	(0.0105905483836510f64, 0.9853540840480058f64),
	(0.0067597991957454f64, 0.9940319694320907f64),
	(0.0029086225531551f64, 0.9988664044200710f64),
];

// 5 of 10 symmetric (weight, abscissa)
const GAUSS_TAB_5: [(f64, f64); 5] = [
	(0.2955242247147529f64, 0.1488743389816312f64),
	(0.2692667193099963f64, 0.4333953941292472f64),
	(0.2190863625159820f64, 0.6794095682990244f64),
	(0.1494513491505806f64, 0.8650633666889845f64),
	(0.0666713443086881f64, 0.9739065285171717f64),
];

macro_rules! quad {
	($tab:expr, $f:expr, $a:expr, $b:expr) => {{
		let rat = ($b-$a)*0.5f64;
		let del = ($a+$b)*0.5f64;
		let mut tot = 0f64;
		for &(w, x) in $tab.iter() {
			tot = w.mul_add(
				$f.eval(rat.mul_add(-x, del)),
				w.mul_add($f.eval(rat.mul_add(x, del)), tot)
			);
		}
		rat * tot
	}};
}

/// Calculates the integral of 'f' from 'a' to 'b'
/// on finite bounds using 10 points of Gaussian 
/// quadrature. See `quad_slow`.
pub fn quad_fast<T: UnivariateFn>(f: &T, a: f64, b: f64) -> f64 {
	quad!(GAUSS_TAB_5, f, a, b)
}

/// Calculates the integral of 'f' from 'a' to 'b'
/// on finite bounds using 50 points of Gaussian
/// quadrature. For functions that can be reasonably
/// approximated by polynomials, this quadrature is
/// extremely accurate. However, it may not be accurate
/// for poorly-behaved functions.
///
/// # Example
/// ```
/// use calc::calculus::*;
///
/// // the integral of x^2 from 0 to 2 is 8/3
/// assert!((quad_slow(&|x: f64| x*x, 0f64, 2f64) - 8f64/3f64).abs() < 1e-7f64);
/// ```
pub fn quad_slow<T: UnivariateFn>(f: &T, a: f64, b: f64) -> f64 {
	quad!(GAUSS_TAB_25, f, a, b)
}

fn trap<'a, T: UnivariateFn>(f: &'a T, a: f64, b: f64, n: usize) -> f64 {
	let beg = f.eval(a);
	let end = f.eval(b);
	match n {
		1 => (b - a) * (end - beg) * 0.5f64,
		_ => {
			let h = (b-a)/(n as f64);
			let mut middle = 0f64;
			let mut pos = a;
			for _ in 1..n {
				pos += h;
				middle += f.eval(pos);
			}
			(beg + middle.mul_add(2f64, end)) * (h * 0.5f64)
		}
	}
}

/// Computes the integral from `a` to `b` of the funtion `f` using
/// Romberg integration. If the integration cannot converge to an
/// accuracy better than `acc`, it returns `None`.
pub fn romberg_integral<T: UnivariateFn>(f: &T, a: f64, b: f64, acc: f64) -> Option<f64> {
	assert!(!acc.is_nan() && !acc.is_infinite() && acc > 0f64);
	const MAX_K: usize = 8;
	let mut tab = [a; MAX_K];
	let mut last;

	for k in 0..MAX_K {
		last = tab[0];
		tab[0] = trap(f, a, b, 1<<k);
		for i in 1..(k+1) {
			let m = 1<<(2*i);
			let next = ((m as f64)*tab[i-1] - last) / ((m-1) as f64);
			
			// check of convergence at the end of the
			// extrapolation series. bailing out earlier
			// doesn't save much time, and forgoes 'cheap'
			// precision.
			if i == k && (last - next).abs() < acc {
				return Some(next);
			}

			tab[k] = next;
			last = tab[i];
			tab[i] = next;
		}
	}
	None
}

/// `BoundsTransform` is a wrapper for univariate
/// functions that allows them to be integrated
/// over non-finite bounds. Note that methods
/// for `BoundsTranform` are less numerically 
/// precise than their ordinary counterparts,
/// all else being equal, so `BoundsTranform`
/// should only be used when one (or both)
/// of the bounds is +/- infinity.
pub struct BoundsTransform<'a, T: 'a> {
	child: &'a T,
}

/// `transform_bounds` returns a struct that
/// can integrate the given function over non-finite
/// bounds.
///
/// # Example
/// ```
/// use calc::calculus::*;
/// use std::f64;
/// 
/// // the integral of e^(-x) from 0 to infinity is 1
/// assert!((transform_bounds(&|x: f64| (-x).exp()).quad_slow(0f64, f64::INFINITY) - 1f64).abs() < 1e-8f64);
/// ```
pub fn transform_bounds<'a, T: UnivariateFn>(f: &'a T) -> BoundsTransform<'a, T> {
	BoundsTransform{ child: f }
}

// a function that maps to (-infinity, +infinity)
// on the interval (-1, 1)
#[inline]
fn xtrans(z: f64) -> f64 {
	-z / ((z-1f64)*(z+1f64))
}

impl<'a, T> UnivariateFn for BoundsTransform<'a, T> where T: UnivariateFn {
	fn eval(&self, x: f64) -> f64 {
		let zsq = x*x;
		let zsm = zsq-1f64;
		self.child.eval(xtrans(x)) * (zsq + 1f64) / (zsm * zsm)
	}
}

#[inline]
fn btrans(a: f64, b: f64) -> (f64, f64) {
	(match a {
		0f64 => 0f64,
		f64::NEG_INFINITY => -1f64 + 1e-10f64,  // close to -1
		_ => ((4f64*a*a+1f64).sqrt()-1f64)/(2f64*a)
	},
	match b {
		0f64 => 0f64,
		f64::INFINITY => 1f64 - 1e-10f64, 		// close to 1
		_ => ((4f64*b*b+1f64).sqrt()-1f64)/(2f64*b)
	})
}

impl<'a, T> BoundsTransform<'a, T> where T: 'a + UnivariateFn {
	/// `quad_fast` calculates the integral
	/// of the tranformed bounds using 10 points
	/// of Gaussian quadrature.
	pub fn quad_fast(&self, a: f64, b: f64) -> f64 {
		if b < a {
			return -self.quad_fast(b, a);
		}
		let (x, y) = btrans(a, b);
		quad_fast(self, x, y)
	}

	/// `quad_slow` calculates the integral of the
	/// transformed bounds using 50 points of 
	/// Gaussian guadrature.
	pub fn quad_slow(&self, a: f64, b: f64) -> f64 {
		if b < a {
			return -self.quad_slow(b, a);
		}
		let (x, y) = btrans(a, b);
		quad_slow(self, x, y)
	}

	/// `romberg_integral` calculates the integral
	/// of the transformed bounds using Romberg
	/// integration.
	pub fn romberg_integral(&self, a: f64, b: f64, acc: f64) -> Option<f64> {
		if b < a {
			if let Some(x) = self.romberg_integral(b, a, acc) {
				return Some(-x);
			}
			return None;
		}
		let (x, y) = btrans(a, b);
		romberg_integral(self, x, y, acc)
	}

}

/*
// reverse-quadratic interpolation
fn rev_quadratic<T: UnivariateFn>(f: &T, a: f64, b: f64, c: f64) -> f64 {
	let fa = f.eval(a);
	let fb = f.eval(b);
	let fc = f.eval(c);
	let s1 = (fb * fc) / ((fa - fb) * (fa - fc));
	let s2 = (fa * fc) / ((fb - fa) * (fb - fc));
	let s3 = (fa * fb) / ((fc - fa) * (fc - fb));
	s1 + s2 + s3
}

// check for convergence of quadratic interpolants
fn check_converge(a: f64, b: f64, c: f64, d: f64, s: f64, flag: bool) -> bool {
	let a_rat = a.mul_add(3f64, b)/4f64;
	if (s > b && s < a_rat && b > a_rat) || (s < b && s > a_rat && b < a_rat) {
		return true;
	}
	if flag {
		if (s-b).abs() >= (b-c).abs()/2f64 || (b-c).abs() < 10e-10f64 {
			return true;
		}
	} else if (s-b).abs() >= (c-d).abs()/2f64 || (c-d).abs() < 10e-10f64 {
		return true;
	}
	false
}


/// `find_root` finds a a point `x` such that `f.eval(x).abs() < acc`.
/// This function asserts that `f.eval(min)` and `f.eval(max)` have
/// opposite signs, and that `acc` is greater than zero. `min` and `max`
/// are simply bounds on the search space of the function.
pub fn find_root<T: UnivariateFn>(f: &T, min: f64, max: f64, acc: f64) -> f64 {
	assert!(acc > 0f64);
	let mut fa = f.eval(min);
	let mut fb = f.eval(max);

	// f(a) and f(b) must have opposite signs
	assert!((fa*fb).is_sign_negative());

	let (mut a, mut b) = (min, max);
	if fa.abs() < fb.abs() {
		(a, b) = (b, a);
	}
	let mut s = 0f64;
	let mut d = a;
	let mut c = a;
	let mut flag = true;
	while (b-a).abs() > acc {
		if a != c && b != c {
			s = rev_quadratic(f, a, b, c);
		} else {
			fb = f.eval(b);
			s = b - fb*(b-a)/(fb - f.eval(a));
		}
		if check_converge(a, b, c, d, s, flag) {
			s = (a + b) / 2f64;
			flag = true;
		} else {
			flag = false;
		}
		(d, c) = (c, b);
		if f.eval(a)*f.eval(s) < 0f64 {
			b = s;
		} else {
			a = s;
		}
		if f.eval(a).abs() < f.eval(b).abs() {
			(a, b) = (b, a);
		}
	}
	s
}*/

#[cfg(test)]
mod tests {
	use std::f64;
	use super::*;
	fn isapprox(a: f64, b: f64) -> bool {
		(a-b).abs() < 1e-8f64
	}

	fn square(x: f64) -> f64 { x*x }

	fn square_integral(a: f64, b: f64) -> f64 {
		(b*b*b)/3f64 - (a*a*a)/3f64
	}

	fn square_diff(x: f64) -> f64 { 2f64*x }

	macro_rules! test_square {
		($a:expr, $b:expr) => {
			{
				want!(quad_slow(&square, $a, $b), square_integral($a, $b));
				want!(quad_fast(&square, $a, $b), square_integral($a, $b));
				want!(romberg_integral(&square, $a, $b, 1e-8f64).unwrap(), square_integral($a, $b));
				want!(diff(&square, $a).unwrap(), square_diff($a));
				want!(diff(&square, $b).unwrap(), square_diff($b));
			}
		};
	}

	macro_rules! test_infinite {
		($a:expr) => {
			{
				let ex = |x: f64| (-x).exp();
				let bt = transform_bounds(&ex);
				want!(bt.quad_slow($a, f64::INFINITY), (-$a).exp());
				want!(bt.quad_fast($a, f64::INFINITY), (-$a).exp(), 1e-4f64);
				want!(bt.romberg_integral($a, f64::INFINITY, 1e-6f64).unwrap(), (-$a).exp(), 1e-6f64);
			}
		};
	}

	macro_rules! want {
		// w/o explicit accuracy
		($a:expr, $b:expr) => {
			{
				let av = $a;
				let bv = $b;
				if !isapprox(av, bv) {
					println!(concat!("case ", stringify!($a), ":"));
					println!("expected {:?}; got {:?}", av, bv);
				}
			}
		};
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
	fn test_integrals() {
		test_square!(0f64, 2f64);
		test_square!(-1f64, 1f64);
		test_square!(1f64, -1f64);
		test_square!(0f64, 27.90821478f64);
		assert!(diff(&square, f64::NAN).is_none());
		assert!(diff(&square, f64::INFINITY).is_none());
	}

	#[test]
	fn test_infinite_integrals() {
		test_infinite!(1f64);
	}
}