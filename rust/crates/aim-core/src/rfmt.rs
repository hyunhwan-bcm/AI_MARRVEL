//! R's printing of doubles as `write.table` does it: `formatReal` with 15 significant digits,
//! fixed notation unless scientific is narrower (`scipen = 0`). Fixed-notation values of 1e15
//! and above (not in AIM's outputs) are printed exactly here, where R prints its own digits.

const DIGITS: i32 = 15; // write.table sets R_print.digits = DBL_DIG
const KP_MAX: i32 = 22;
const R_DEC_MIN_EXPONENT: i32 = -308;

fn tbl(k: i32) -> f64 {
    10f64.powi(k) // exact for 0..=22
}

/// `scientific()` from R 4.4 `src/main/format.c`, the branch for platforms where long double
/// is no wider than double (arm64 macOS, where the goldens come from): |x| scaled to 15 digits
/// in double arithmetic, rounded half-even, trailing zeros dropped. That scaling can round a
/// last digit away that correct rounding keeps (0.3239888683847305 prints with 14 digits).
/// On x86-64 Linux R scales in 80-bit long double, so such last digits can differ there.
/// Returns (kpower, nsig, roundingwidens).
fn scientific(r: f64) -> (i32, i32, bool) {
    let mut kp = r.log10().floor() as i32 - DIGITS + 1;
    let mut r_prec = r;
    if kp.abs() <= KP_MAX {
        if kp >= 0 {
            r_prec /= tbl(kp);
        } else {
            r_prec *= tbl(-kp);
        }
    } else if kp <= R_DEC_MIN_EXPONENT {
        r_prec = (r_prec * 1e303) / 10f64.powf((kp + 303) as f64);
    } else {
        r_prec /= 10f64.powf(kp as f64);
    }
    if r_prec < tbl(DIGITS - 1) {
        r_prec *= 10.0;
        kp -= 1;
    }
    let mut alpha = r_prec.round_ties_even(); // nearbyint
    let mut nsig = DIGITS;
    for _ in 1..=DIGITS {
        alpha /= 10.0;
        if alpha == alpha.floor() {
            nsig -= 1;
        } else {
            break;
        }
    }
    if nsig == 0 {
        nsig = 1;
        kp += 1;
    }
    let kpower = kp + DIGITS - 1;
    let rgt = (DIGITS - kpower).clamp(0, KP_MAX);
    let fuzz = 0.5 / tbl(rgt);
    let widens = kpower > 0 && kpower <= KP_MAX && r < tbl(kpower) - fuzz;
    (kpower, nsig, widens)
}

/// One double as `write.table` writes it (`formatReal` + `EncodeReal0` for a single value;
/// `NA` for a missing value).
pub fn format_real(x: Option<f64>) -> String {
    let Some(x) = x else {
        return "NA".into();
    };
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf" } else { "-Inf" }.into();
    }
    let neg = i32::from(x < 0.0);
    let (kpower, nsig, widens) = if x == 0.0 {
        (0, 1, false)
    } else {
        scientific(x.abs())
    };
    let left = kpower + 1 - i32::from(widens);
    let mut sleft = neg + if left <= 0 { 1 } else { left };
    let rgt = (nsig - left).max(0);
    if left < 0 {
        sleft = 1 + neg;
    }
    let width_fixed = sleft + rgt + i32::from(rgt != 0);
    let e = if left > 100 || left <= -99 { 2 } else { 1 };
    let d = nsig - 1;
    let width_sci = neg + i32::from(d > 0) + d + 4 + e;
    if width_fixed <= width_sci {
        format!("{:.*}", rgt as usize, x)
    } else {
        let s = format!("{:.*e}", d as usize, x);
        let (m, exp) = s.split_once('e').unwrap();
        let exp: i32 = exp.parse().unwrap();
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_like_r() {
        let f = |x: f64| format_real(Some(x));
        assert_eq!(f(1.0), "1");
        assert_eq!(f(612367.0), "612367");
        assert_eq!(f(100196914.0), "100196914");
        assert_eq!(f(0.848117493450182), "0.848117493450182");
        assert_eq!(f(0.1 + 0.2), "0.3");
        assert_eq!(f(0.0001), "1e-04");
        assert_eq!(f(0.00012), "0.00012");
        assert_eq!(f(1e15), "1e+15");
        assert_eq!(f(123456.5), "123456.5");
        assert_eq!(f(-2.5), "-2.5");
        assert_eq!(f(1.0 / 3.0), "0.333333333333333");
        assert_eq!(f(2.0 / 3.0 * 1e-5), "6.66666666666667e-06");
        assert_eq!(f(0.3239888683847305), "0.32398886838473"); // R's scaling, not correct rounding
        assert_eq!(f(0.0), "0");
        assert_eq!(format_real(None), "NA");
    }
}
