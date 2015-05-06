//! The `calc` crate provides functions and traits related
//! to numerical methods.
//!
//!

/// UnivariateFn is the trait that
/// is implemented by simple univariate functions.
pub trait UnivariateFn {
	fn eval(&self, f64) -> f64;
}

// 50 points of Gauss-Legendre weights and abscissae
const GAUSS_TAB_50: [(f64, f64); 50] = [
	(0.0621766166553473f64, -0.0310983383271889f64),
	(0.0621766166553473f64, 0.0310983383271889f64),
	(0.0619360674206832f64, -0.0931747015600861f64),
	(0.0619360674206832f64, 0.0931747015600861f64),
	(0.0614558995903167f64, -0.1548905899981459f64),
	(0.0614558995903167f64, 0.1548905899981459f64),
	(0.0607379708417702f64, -0.2160072368760418f64),
	(0.0607379708417702f64, 0.2160072368760418f64),
	(0.0597850587042655f64, -0.2762881937795320f64),
	(0.0597850587042655f64, 0.2762881937795320f64),
	(0.0586008498132224f64, -0.3355002454194373f64),
	(0.0586008498132224f64, 0.3355002454194373f64),
	(0.0571899256477284f64, -0.3934143118975651f64),
	(0.0571899256477284f64, 0.3934143118975651f64),
	(0.0555577448062125f64, -0.4498063349740388f64),
	(0.0555577448062125f64, 0.4498063349740388f64),
	(0.0537106218889962f64, -0.5044581449074642f64),
	(0.0537106218889962f64, 0.5044581449074642f64),
	(0.0516557030695811f64, -0.5571583045146501f64),
	(0.0516557030695811f64, 0.5571583045146501f64),
	(0.0494009384494663f64, -0.6077029271849502f64),
	(0.0494009384494663f64, 0.6077029271849502f64),
	(0.0469550513039484f64, -0.6558964656854394f64),
	(0.0469550513039484f64, 0.6558964656854394f64),
	(0.0443275043388033f64, -0.7015524687068222f64),
	(0.0443275043388033f64, 0.7015524687068222f64),
	(0.0415284630901477f64, -0.7444943022260685f64),	
	(0.0415284630901477f64, 0.7444943022260685f64),
	(0.0385687566125877f64, -0.7845558329003993f64),
	(0.0385687566125877f64, 0.7845558329003993f64),
	(0.0354598356151462f64, -0.8215820708593360f64),
	(0.0354598356151462f64, 0.8215820708593360f64),
	(0.0322137282235780f64, -0.8554297694299461f64),
	(0.0322137282235780f64, 0.8554297694299461f64),
	(0.0288429935805352f64, -0.8859679795236131f64),
	(0.0288429935805352f64, 0.8859679795236131f64),
	(0.0253606735700124f64, -0.9130785566557919f64),
	(0.0253606735700124f64, 0.9130785566557919f64),
	(0.0217802431701248f64, -0.9366566189448780f64),
	(0.0217802431701248f64, 0.9366566189448780f64),
	(0.0181155607134894f64, -0.9566109552428079f64),
	(0.0181155607134894f64, 0.9566109552428079f64),
	(0.0143808227614856f64, -0.9728643851066920f64),
	(0.0143808227614856f64, 0.9728643851066920f64),
	(0.0105905483836510f64, -0.9853540840480058f64),
	(0.0105905483836510f64, 0.9853540840480058f64),
	(0.0067597991957454f64, -0.9940319694320907f64),
	(0.0067597991957454f64, 0.9940319694320907f64),
	(0.0029086225531551f64, -0.9988664044200710f64),
	(0.0029086225531551f64, 0.9988664044200710f64),
];

// 10 points of gauss quadrature 
const GAUSS_TAB_10: [(f64, f64); 10] = [
	(0.2955242247147529f64, -0.1488743389816312f64),
	(0.2955242247147529f64, 0.1488743389816312f64),
	(0.2692667193099963f64, -0.4333953941292472f64),
	(0.2692667193099963f64, 0.4333953941292472f64),
	(0.2190863625159820f64, -0.6794095682990244f64),
	(0.2190863625159820f64, 0.6794095682990244f64),
	(0.1494513491505806f64, -0.8650633666889845f64),
	(0.1494513491505806f64, 0.8650633666889845f64),
	(0.0666713443086881f64, -0.9739065285171717f64),
	(0.0666713443086881f64, 0.9739065285171717f64),
];

macro_rules! quad {
	($tab:expr, $f:expr, $a:expr, $b:expr) => {{
		let rat = ($b-$a)/2f64;
		let del = ($a+$b)/2f64;
		let mut tot = 0f64;
		for &(w, x) in $tab.iter() {
			tot = w.mul_add($f.eval(rat.mul_add(x, del)), tot);
		}
		rat * tot
	}};
}

/// Calculates the integral of 'f' from 'a' to 'b'
/// on finite bounds using 10 points of Gaussian 
/// quadrature. See `quad_slow`.
pub fn quad_fast<T: UnivariateFn>(f: &T, a: f64, b: f64) -> f64 {
	quad!(GAUSS_TAB_10, f, a, b)
}

/// Calculates the integral of 'f' from 'a' to 'b'
/// on finite bounds using 50 points of Gaussian
/// quadrature.
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
	quad!(GAUSS_TAB_50, f, a, b)
}

// 1st-order central diff approximation
#[inline]
fn fa0<T: UnivariateFn>(f: &T, x: f64, h: f64) -> f64 {
	(f.eval(x+h)-f.eval(x-h))/(2f64*h)
}

// 2nd-order central diff approximation
#[inline]
fn fa1<T: UnivariateFn>(f: &T, x: f64, h: f64) -> f64 {
	(4f64*fa0(f, x, h/2f64) - fa0(f, x, h))/3f64
}

// 3rd-order central diff approximation
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
/// 
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

#[cfg(test)]
mod tests {
	use super::*;

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
		(a-b).abs() < 1e-10f64
	}

	macro_rules! test_square {
		($a:expr, $b:expr) => {
			{
				want!(quad_slow(&Square, $a, $b), square_integral($a, $b));
				want!(quad_slow(&Square, $a, $b), square_integral($a, $b));
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
	}
}


