//! Ranking and confidence as computed in `bin/extraModel_main.py` and
//! `bin/extraModel/confidence.py` (scipy 1.10 semantics).

/// `scipy.stats.percentileofscore(reference, score, kind="rank")`.
pub fn percentile_of_score(reference: &[f64], score: f64) -> f64 {
    let left = reference.iter().filter(|&&a| a < score).count();
    let right = reference.iter().filter(|&&a| a <= score).count();
    let plus1 = usize::from(left < right);
    (left + right + plus1) as f64 * (50.0 / reference.len() as f64)
}

/// Confidence category from a confidence score, as `assign_confidence_score`.
pub fn confidence_level(confidence: f64) -> &'static str {
    if confidence < 25.0 {
        "Unsolved"
    } else if confidence < 50.0 {
        "Solved (Low)"
    } else if confidence < 75.0 {
        "Solved (Medium)"
    } else {
        "Solved (High)"
    }
}

/// `min_ranking` and `max_ranking` from `assign_ranking`: ranks of `1 - predict` computed in
/// `f32` (numpy keeps the float32 dtype), so predictions that differ only below f32 precision
/// near 1.0 tie, exactly as in the pipeline. Returns 1-based (min, max) rank per input.
pub fn rank_predictions(predict: &[f32]) -> Vec<(usize, usize)> {
    let keys: Vec<f32> = predict.iter().map(|&p| 1.0f32 - p).collect();
    let mut sorted = keys.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    keys.iter()
        .map(|k| {
            let below = sorted.partition_point(|s| s < k);
            let at_or_below = sorted.partition_point(|s| s <= k);
            (below + 1, at_or_below)
        })
        .collect()
}

/// `scipy.stats.rankdata(values, method="max")`: 1-based rank, ties share the highest rank.
pub fn rankdata_max(values: &[f64]) -> Vec<usize> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    values
        .iter()
        .map(|v| sorted.partition_point(|s| s <= v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_scipy_rank_kind() {
        // scipy.stats.percentileofscore([1, 2, 3, 4], 3) == 75.0
        assert_eq!(percentile_of_score(&[1.0, 2.0, 3.0, 4.0], 3.0), 75.0);
        // scipy.stats.percentileofscore([1, 2, 3, 3, 4], 3) == 70.0
        assert_eq!(percentile_of_score(&[1.0, 2.0, 3.0, 3.0, 4.0], 3.0), 70.0);
        assert_eq!(percentile_of_score(&[1.0, 2.0], 0.5), 0.0);
    }

    #[test]
    fn rankdata_max_matches_scipy() {
        // scipy.stats.rankdata([0.0, 2.0, 0.0, 1.0], "max") == [2, 4, 2, 3]
        assert_eq!(rankdata_max(&[0.0, 2.0, 0.0, 1.0]), vec![2, 4, 2, 3]);
    }

    #[test]
    fn ties_share_min_and_max_rank() {
        // 1 - p: [0.1, 0.5, 0.5, 0.9] -> min ranks [1, 2, 2, 4], max ranks [1, 3, 3, 4]
        let r = rank_predictions(&[0.9, 0.5, 0.5, 0.1]);
        assert_eq!(r, vec![(1, 1), (2, 3), (2, 3), (4, 4)]);
    }
}
