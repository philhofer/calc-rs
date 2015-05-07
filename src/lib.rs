//! The `calc` crate provides functions and traits related
//! to numerical methods.
//!
//!

/// UnivariateFn is the trait that
/// is implemented by simple univariate functions.
pub trait UnivariateFn {
	fn eval(&self, f64) -> f64;
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
		$x.mul_add(poly!($x; $($b),*), $a)
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
/// quadrature. For functions that can be accurately
/// approximated by polynomials, this quadrature is
/// extremely accurate. However, it may not be accurate
/// for poorly-behaved functions.
///
/// # Examples
///
/// ```
/// use calc::*;
///
/// struct Square;
/// 
/// impl UnivariateFn for Square {
///		fn eval(&self, x: f64) -> f64 { x*x }
/// }
///
/// // the integral of x^2 from 0 to 2 is 8/3
/// assert!((quad_slow(&Square, 0f64, 2f64) - 8f64/3f64).abs() < 1e-8f64);
/// ```
pub fn quad_slow<T: UnivariateFn>(f: &T, a: f64, b: f64) -> f64 {
	quad!(GAUSS_TAB_25, f, a, b)
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
/// # Examples
///
/// ```
/// use calc::*;
/// struct Square;
/// impl UnivariateFn for Square {
///		fn eval(&self, x: f64) -> f64 { x*x }
/// }
///
/// // the derivate of x^2 at 1 is 2
/// assert!((diff(&Square, 1f64).unwrap() - 2f64).abs() < 1e-10f64);
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

fn trap<'a, T: UnivariateFn>(f: &'a T, a: f64, b: f64, n: usize) -> f64 {
	let beg = f.eval(a);
	let end = f.eval(b);
	match n {
		0 => a,
		1 => (b - a) * (end - beg) * 0.5f64,
		_ => {
			let h = (b-a)/(n as f64);
			let mut middle = 0f64;
			let mut pos = a;
			for _ in 1..n {
				pos += h;
				middle += f.eval(pos);
			}
			(beg + 2f64*middle + end) * (h*0.5f64)
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
	let mut last = 0f64;
	for k in 1..MAX_K {
		for i in 1..(k+1) {
			match i {
				1 => {
					last = tab[i];
					tab[i] = trap(f, a, b, 1<<(k-1));
				},
				_ => {
					let m = 1<<((i-1)*2);
					let next = ((m as f64)*tab[i-1] - last) / ((m-1) as f64);
					if (last - next).abs() < acc {
						return Some(next);
					}
					tab[k] = next;
					last = tab[i];
					tab[i] = tab[k];
				}
			}
		}
	}
	None
}
