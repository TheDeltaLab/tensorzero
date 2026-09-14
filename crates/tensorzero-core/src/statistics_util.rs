//! Statistical utility functions.

/// Computes the mean of a slice of data.
pub fn mean(data: &[f32]) -> Option<f32> {
    let sum = data.iter().sum::<f32>();
    let count = data.len();

    if count == 0 {
        return None;
    }

    Some(sum / count as f32)
}

/// Computes the standard deviation of a slice of data.
pub fn std_deviation(data: &[f32]) -> Option<f32> {
    let count = data.len();
    if count == 0 {
        return None;
    }
    let data_mean = mean(data)?;

    let variance = data
        .iter()
        .map(|value| {
            let diff = data_mean - (*value);
            diff * diff
        })
        .sum::<f32>()
        / count as f32;

    Some(variance.sqrt())
}

/// Computes the Wald confidence interval for continuous data.
/// Uses the formula: mean ± 1.96 * (stddev / sqrt(n))
/// Returns (lower, upper) bounds, or (None, None) if insufficient data.
pub fn wald_confint(mean: f64, stdev: f64, count: u32) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }

    let count_f64 = count as f64;
    let margin = 1.96 * (stdev / count_f64.sqrt());
    Some((mean - margin, mean + margin))
}

/// Computes the 95% Wilson score confidence interval for Bernoulli data.
///
/// This function takes precomputed mean (proportion) and count, and returns
/// the lower and upper bounds of the confidence interval.
///
/// The Wilson score interval is preferred over the Wald interval for proportions
/// because it handles extreme cases (p close to 0 or 1) better and doesn't produce
/// bounds outside [0, 1].
///
/// # Arguments
/// * `mean` - The proportion/mean of successes (should be in [0, 1] for Bernoulli data)
/// * `count` - The number of observations
///
/// # Returns
/// * `Some((lower, upper))` if count > 0
/// * `None` if count is 0
pub fn wilson_confint(mean: f64, count: u32) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }

    let count_f64 = count as f64;
    let z: f64 = 1.96; // Standard normal quantile for 95% CI
    let z_squared: f64 = z.powi(2);
    let scale: f64 = 1.0 / (1.0 + z_squared / count_f64);
    let center: f64 = mean + z_squared / (2.0 * count_f64);
    let margin = z / (2.0 * count_f64) * (4.0 * count_f64 * mean * (1.0 - mean) + z_squared).sqrt();
    let ci_lower = (center - margin) * scale;
    let ci_upper = (center + margin) * scale;

    // Clamp to [0, 1] to handle floating-point precision errors
    let ci_lower = ci_lower.clamp(0.0, 1.0);
    let ci_upper = ci_upper.clamp(0.0, 1.0);

    Some((ci_lower, ci_upper))
}

/// Computes the 95% Wilson score confidence interval from raw Bernoulli data.
///
/// This is a convenience function that computes the mean from the data and
/// then calls `wilson_confint`.
///
/// # Arguments
/// * `data` - Slice of values (typically 0.0 or 1.0 for Bernoulli data)
///
/// # Returns
/// * `Some((lower, upper))` if data is non-empty
/// * `None` if data is empty
pub fn wilson_confint_from_data(data: &[f32]) -> Option<(f64, f64)> {
    let count = data.len();
    if count == 0 {
        return None;
    }

    let sum: f32 = data.iter().sum();
    let mean = (sum / count as f32) as f64;

    wilson_confint(mean, count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for wald confidence interval
    #[test]
    fn test_wald_confint_basic() {
        // Test basic Wald CI computation
        // mean = 100, stdev = 10, count = 100
        // margin = 1.96 * (10 / sqrt(100)) = 1.96
        let (lower, upper) =
            wald_confint(100.0, 10.0, 100).expect("Should produce valid confidence interval");
        // 100 - 1.96 = 98.04, 100 + 1.96 = 101.96
        assert!((lower - 98.04).abs() < 0.01);
        assert!((upper - 101.96).abs() < 0.01);
    }

    #[test]
    fn test_wald_confint_zero_count() {
        // Test with zero count
        let confint = wald_confint(100.0, 10.0, 0);
        assert!(
            confint.is_none(),
            "Zero count should produce None as confidence interval"
        );
    }

    // Tests for wilson confidence interval

    #[test]
    fn test_wilson_confint_basic() {
        // Test Wilson CI for p = 0.5, n = 100
        // This should give approximately (0.40, 0.60)
        let result = wilson_confint(0.5, 100);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        // Wilson CI for p=0.5, n=100 should be approximately (0.40, 0.60)
        assert!(lower > 0.39 && lower < 0.41, "lower = {lower}");
        assert!(upper > 0.59 && upper < 0.61, "upper = {upper}");
    }

    #[test]
    fn test_wilson_confint_extreme_high() {
        // Test Wilson CI for p = 1.0, n = 1 (extreme case)
        let result = wilson_confint(1.0, 1);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        // Lower should be around 0.206
        assert!(
            (lower - 0.20654329147389294).abs() < 0.0001,
            "lower = {lower}"
        );
        assert!((upper - 1.0).abs() < 0.0001, "upper = {upper}");
    }

    #[test]
    fn test_wilson_confint_zero_count() {
        let result = wilson_confint(0.5, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_wilson_confint_known_values() {
        // Test against known values: p = 0.4489795918367347, n = 49
        let result = wilson_confint(0.4489795918367347, 49);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        assert!(
            (lower - 0.31852624929636336).abs() < 0.0001,
            "lower = {lower}"
        );
        assert!(
            (upper - 0.5868513320032188).abs() < 0.0001,
            "upper = {upper}"
        );
    }

    // Tests for wilson_confint_from_data (takes raw data)

    #[test]
    fn test_wilson_confint_from_data_all_zeros() {
        // All 0s should still produce non-zero width interval
        let data = vec![0.0; 10];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower == 0.0, "Lower bound should be = 0, got {lower}");
        assert!(upper < 1.0, "Upper bound should be < 1, got {upper}");

        // Should have non-zero width
        assert!(
            upper > lower,
            "Interval should have non-zero width, got [{lower}, {upper}]"
        );
        assert!(
            upper > 0.0,
            "Upper bound should be > 0 even with all zeros, got {upper}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_all_ones() {
        // All 1s should still produce non-zero width interval
        let data = vec![1.0; 10];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower > 0.0, "Lower bound should be > 0, got {lower}");
        assert!(upper == 1.0, "Upper bound should be = 1, got {upper}");

        // Should have non-zero width
        assert!(
            upper > lower,
            "Interval should have non-zero width, got [{lower}, {upper}]"
        );
        assert!(
            lower < 1.0,
            "Lower bound should be < 1 even with all ones, got {lower}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_half_half() {
        // 50/50 split should have symmetric interval around 0.5
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower > 0.0, "Lower bound should be > 0, got {lower}");
        assert!(upper < 1.0, "Upper bound should be < 1, got {upper}");

        // Should be roughly symmetric around 0.5
        let midpoint = (lower + upper) / 2.0;
        assert!(
            (midpoint - 0.5).abs() < 0.01,
            "Midpoint should be close to 0.5, got {midpoint}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_single_value() {
        // Single observation should still work
        let data = vec![1.0];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
        assert!(upper > lower);
    }

    #[test]
    fn test_wilson_confint_from_data_empty() {
        // Empty data should return None
        let data: Vec<f32> = vec![];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_none());
    }

    #[test]
    fn test_wilson_confint_from_data_known_values() {
        // Test against known values
        let data = vec![0.0; 10];
        let mut test_data = data;
        test_data.extend(vec![1.0; 10]);

        let result = wilson_confint_from_data(&test_data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        assert!(
            (lower - 0.2992949).abs() < 1e-6,
            "Lower bound should be approximately 0.2992949, got {lower}"
        );
        assert!(
            (upper - 0.7007050).abs() < 1e-6,
            "Upper bound should be approximately 0.7007050, got {upper}"
        );
    }

    #[test]
    fn test_wilson_vs_wald_large_sample() {
        // With large samples, Wilson and Wald intervals should converge
        // Using n=10000, p=0.4 (4000 successes)
        let mut data = vec![0.0; 6000];
        data.extend(vec![1.0; 4000]);

        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (wilson_lower, wilson_upper) = result.unwrap();

        // Compute Wald interval: p̂ ± 1.96 * sqrt(p̂(1-p̂)/n)
        let p_hat: f64 = 0.4;
        let n: f64 = 10000.0;
        let wald_stderr: f64 = (p_hat * (1.0 - p_hat) / n).sqrt();
        let wald_margin: f64 = 1.96 * wald_stderr;
        let wald_lower: f64 = p_hat - wald_margin;
        let wald_upper: f64 = p_hat + wald_margin;

        // Wilson and Wald should be very close for large n
        assert!(
            (wilson_lower - wald_lower).abs() < 1e-4,
            "Wilson lower ({wilson_lower}) should be close to Wald lower ({wald_lower})"
        );
        assert!(
            (wilson_upper - wald_upper).abs() < 1e-4,
            "Wilson upper ({wilson_upper}) should be close to Wald upper ({wald_upper})"
        );

        // Verify both are well within [0, 1]
        assert!(wilson_lower > 0.0 && wilson_lower < 1.0);
        assert!(wilson_upper > 0.0 && wilson_upper < 1.0);
        assert!(wald_lower > 0.0 && wald_lower < 1.0);
        assert!(wald_upper > 0.0 && wald_upper < 1.0);
    }

    #[test]
    fn test_wilson_confint_from_data_versus_wald_extreme() {
        // Wilson should handle extreme proportions better than Wald
        // With p=0.05 (1 success in 20), Wald can give lower < 0
        let mut data = vec![0.0; 19];
        data.push(1.0);

        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, _upper) = result.unwrap();

        // Wilson should keep bounds within [0, 1]
        assert!(
            lower >= 0.0,
            "Wilson lower bound should be >= 0, got {lower}",
        );
    }
}

use crate::{
    db::feedback::{CumulativeFeedbackTimeSeriesPoint, InternalCumulativeFeedbackTimeSeriesPoint},
    error::{Error, ErrorDetails},
};

/// Computes asymptotic confidence sequences for the running average conditional mean.
///
/// This function computes an asymptotic confidence sequence for the running average
/// conditional mean that is valid under martingale dependence. If the data are i.i.d.,
/// then this returns an asymptotic confidence sequence for the mean. See the referencee
/// below for the definition of an asymptotic confidence sequence and the expression that
/// this function implements (Proposition 2.5).
///
/// # Arguments
///
/// * `feedback` - A vector of time series points, each containing cumulative statistics
///   (mean, variance, count), where count is equivalent to time.
/// * `alpha` - The significance level, in (0, 1). The confidence sequence
///   will have coverage probability 1 - alpha. For example, alpha = 0.05 gives 95% confidence.
/// * `rho` - Optional nonnegative tuning parameter that determines the "intrinsic time" scale. Controls
///   the time point at which the confidence sequence is tightest, relatively speaking. Can be
///   selected to make the confidence sequence is relatively tight at a given time point, if
///   the user anticipates checking the confidence sequence (e.g. for hypothesis testing
///   purposes) starting at or around that time point. If `None`, a value is computed which is
///   approximately optimal in the i.i.d. setting for times starting at 100. The confidence
///   sequence is not highly sensitive to rho, so it's fine to leave it unspecified.
///
/// # Returns
///
/// Returns a `Result` containing a vector of `CumulativeFeedbackTimeSeriesPoint` with confidence
/// sequence bounds (`cs_lower`, `cs_upper`) added to each point. The bounds are symmetric
/// around the mean, with margin computed as:
///
/// ```text
/// margin = sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
/// ```
///
/// where n is the count, v is the variance, and rho is the tuning parameter.
///
/// # Errors
///
/// Returns an error if:
/// * `alpha` is not in the open interval (0, 1)
/// * `rho` is not strictly positive (if provided)
///
/// # References
///
/// Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
/// Time-uniform central limit theory and asymptotic confidence sequences.
/// *The Annals of Statistics*, 52(6), 2380-2407.
/// DOI: [10.1214/24-AOS2408](https://doi.org/10.1214/24-AOS2408)
///
/// # Example
///
/// ```ignore
/// let feedback = vec![
///     InternalCumulativeFeedbackTimeSeriesPoint {
///         period_end: chrono::Utc::now(),
///         variant_name: "control".to_string(),
///         mean: 0.5,
///         variance: 0.25,
///         count: 100,
///     },
/// ];
/// let result = asymp_cs(feedback, 0.05, None)?;
/// // result[0].cs_lower and result[0].cs_upper contain 95% confidence bounds
/// ```
pub fn asymp_cs(
    feedback: Vec<InternalCumulativeFeedbackTimeSeriesPoint>,
    alpha: f32,
    rho: Option<f32>,
) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("alpha must be in (0, 1), got {alpha}"),
        }));
    }

    // Default value of rho, computed as sqrt( (-2 log(alpha) + log(-2 log(alpha)) + 1) / 100 )
    let rho =
        rho.unwrap_or_else(|| (-2.0 * alpha.ln() + (-2.0 * alpha.ln()).ln() + 1.0).sqrt() / 10.0);

    if rho <= 0.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("rho must be strictly positive, got {rho}"),
        }));
    }
    let rho2 = rho * rho;
    let alpha2 = alpha * alpha;

    Ok(feedback
        .into_iter()
        .map(|f| {
            // If variance is None, we can't compute confidence sequences
            let (cs_lower, cs_upper) = match (f.mean, f.variance) {
                (mean, Some(variance)) => {
                    let count_f32 = f.count as f32;
                    let cv_rho2 = count_f32 * variance * rho2;
                    // Compute margin: sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
                    let margin = ((cv_rho2 + 1.0) / (count_f32 * count_f32 * rho2)
                        * ((cv_rho2 + 1.0) / alpha2).ln())
                    .sqrt();
                    (Some(mean - margin), Some(mean + margin))
                }
                _ => (None, None),
            };

            CumulativeFeedbackTimeSeriesPoint {
                period_end: f.period_end,
                variant_name: f.variant_name,
                mean: f.mean,
                variance: f.variance,
                count: f.count,
                alpha,
                cs_lower,
                cs_upper,
            }
        })
        .collect())
}

#[cfg(test)]
mod asymp_cs_tests {
    use super::*;
    use chrono::Utc;

    fn create_test_point(
        mean: f32,
        variance: f32,
        count: u64,
    ) -> InternalCumulativeFeedbackTimeSeriesPoint {
        InternalCumulativeFeedbackTimeSeriesPoint {
            period_end: Utc::now(),
            variant_name: "test_variant".to_string(),
            mean,
            variance: Some(variance),
            count,
        }
    }

    #[test]
    fn test_basic_functionality() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].mean, 0.5);
        assert_eq!(result[0].variance, Some(0.25));
        assert_eq!(result[0].count, 100);
        assert_eq!(result[0].alpha, 0.05);
        assert!(result[0].cs_lower.unwrap() < result[0].mean);
        assert!(result[0].cs_upper.unwrap() > result[0].mean);
    }

    #[test]
    fn test_empty_input() {
        let feedback = vec![];
        let result = asymp_cs(feedback, 0.05, None).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_data_point() {
        let feedback = vec![create_test_point(0.8, 0.16, 50)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 1);
        // Verify bounds are symmetric
        let mean = result[0].mean;
        let cs_lower = result[0].cs_lower.unwrap();
        let cs_upper = result[0].cs_upper.unwrap();
        let margin = cs_upper - mean;
        assert!((mean - cs_lower - margin).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_data_points() {
        let feedback = vec![
            create_test_point(0.3, 0.21, 10),
            create_test_point(0.5, 0.25, 50),
            create_test_point(0.7, 0.21, 100),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 3);
        // Verify all points have confidence sequences
        for point in result {
            assert!(point.cs_lower.unwrap() < point.mean);
            assert!(point.cs_upper.unwrap() > point.mean);
            assert_eq!(point.alpha, 0.05);
        }
    }

    #[test]
    fn test_confidence_sequences_narrow_with_more_data() {
        let feedback = vec![
            create_test_point(0.5, 0.25, 10),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.5, 0.25, 1000),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        let width_10 = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        let width_100 = result[1].cs_upper.unwrap() - result[1].cs_lower.unwrap();
        let width_1000 = result[2].cs_upper.unwrap() - result[2].cs_lower.unwrap();

        // Widths should decrease with more data
        assert!(width_10 > width_100);
        assert!(width_100 > width_1000);
    }

    #[test]
    fn test_higher_variance_wider_intervals() {
        let feedback = vec![
            create_test_point(0.5, 0.1, 100),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.5, 0.5, 100),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        let width_low_var = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        let width_med_var = result[1].cs_upper.unwrap() - result[1].cs_lower.unwrap();
        let width_high_var = result[2].cs_upper.unwrap() - result[2].cs_lower.unwrap();

        // Widths should increase with variance
        assert!(width_low_var < width_med_var);
        assert!(width_med_var < width_high_var);
    }

    #[test]
    fn test_smaller_alpha_wider_intervals() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];

        let result_95 = asymp_cs(feedback.clone(), 0.05, None).unwrap();
        let result_99 = asymp_cs(feedback, 0.01, None).unwrap();

        let width_95 = result_95[0].cs_upper.unwrap() - result_95[0].cs_lower.unwrap();
        let width_99 = result_99[0].cs_upper.unwrap() - result_99[0].cs_lower.unwrap();

        // 99% confidence should be wider than 95%
        assert!(width_99 > width_95);
    }

    #[test]
    fn test_custom_rho() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];

        let result_default = asymp_cs(feedback.clone(), 0.05, None).unwrap();
        let result_custom = asymp_cs(feedback, 0.05, Some(0.5)).unwrap();

        // Different rho values should give different bounds
        assert_ne!(
            result_default[0].cs_lower.unwrap(),
            result_custom[0].cs_lower.unwrap()
        );
        assert_ne!(
            result_default[0].cs_upper.unwrap(),
            result_custom[0].cs_upper.unwrap()
        );
    }

    #[test]
    fn test_bounds_symmetry() {
        let feedback = vec![
            create_test_point(0.2, 0.16, 50),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.8, 0.16, 150),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // Verify bounds are symmetric around the mean
        for point in result {
            let mean = point.mean;
            let cs_lower = point.cs_lower.unwrap();
            let cs_upper = point.cs_upper.unwrap();
            let lower_margin = mean - cs_lower;
            let upper_margin = cs_upper - mean;
            assert!((lower_margin - upper_margin).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zero_variance() {
        let feedback = vec![create_test_point(0.5, 0.0, 100)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // Should still produce valid (narrow) bounds
        assert!(result[0].cs_lower.unwrap() < result[0].mean);
        assert!(result[0].cs_upper.unwrap() > result[0].mean);
        // With zero variance, bounds should be very tight
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width < 0.2);
    }

    #[test]
    fn test_preserves_input_fields() {
        let feedback = vec![InternalCumulativeFeedbackTimeSeriesPoint {
            period_end: Utc::now(),
            variant_name: "variant_a".to_string(),
            mean: 0.42,
            variance: Some(0.24),
            count: 123,
        }];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result[0].variant_name, "variant_a");
        assert_eq!(result[0].mean, 0.42);
        assert_eq!(result[0].variance, Some(0.24));
        assert_eq!(result[0].count, 123);
    }

    // Error cases
    #[test]
    fn test_alpha_zero() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.0, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("alpha must be in (0, 1)")
        );
    }

    #[test]
    fn test_alpha_one() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 1.0, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("alpha must be in (0, 1)")
        );
    }

    #[test]
    fn test_alpha_negative() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, -0.1, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("alpha must be in (0, 1)")
        );
    }

    #[test]
    fn test_alpha_greater_than_one() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 1.5, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("alpha must be in (0, 1)")
        );
    }

    #[test]
    fn test_rho_zero() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, Some(0.0));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("rho must be strictly positive")
        );
    }

    #[test]
    fn test_rho_negative() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, Some(-0.5));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("rho must be strictly positive")
        );
    }

    #[test]
    fn test_margin_calculation_correctness() {
        // Test the actual margin calculation with known values
        let mean = 0.5_f32;
        let variance = 0.25_f32;
        let count = 100_u64;
        let alpha = 0.05_f32;
        let rho = 0.296_f32;

        let feedback = vec![create_test_point(mean, variance, count)];
        let result = asymp_cs(feedback, alpha, Some(rho)).unwrap();

        // Manually compute expected margin
        let count_f32 = count as f32;
        let rho2 = rho * rho;
        let alpha2 = alpha * alpha;
        let cv_rho2 = count_f32 * variance * rho2;
        let expected_margin = ((cv_rho2 + 1.0) / (count_f32 * count_f32 * rho2)
            * ((cv_rho2 + 1.0) / alpha2).ln())
        .sqrt();

        let actual_margin = result[0].cs_upper.unwrap() - result[0].mean;
        assert!((actual_margin - expected_margin).abs() < 1e-5);
    }

    #[test]
    fn test_large_count() {
        let feedback = vec![create_test_point(0.5, 0.25, 1_000_000)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // With very large count, bounds should be very tight
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width < 0.01);
    }

    #[test]
    fn test_small_count() {
        let feedback = vec![create_test_point(0.5, 0.25, 2)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // With small count, bounds should be wide
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width > 0.1);
    }
}
