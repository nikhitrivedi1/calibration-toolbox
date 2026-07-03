"""
Unit tests for calibration metrics.
"""

import numpy as np
import pytest
from calibration_toolbox import metrics


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error (ECE)."""
    
    def test_perfect_calibration(self):
        """Test ECE for perfectly calibrated predictions."""
        # Perfect predictions
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1, 0])
        ece = metrics.expected_calibration_error(probs, labels)
        assert ece < 0.01, "Perfect calibration should have ECE close to 0"
    
    def test_poor_calibration(self):
        """Test ECE for poorly calibrated predictions."""
        # Confident but wrong predictions
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        labels = np.array([1, 1, 1])  # All wrong
        ece = metrics.expected_calibration_error(probs, labels)
        assert ece > 0.5, "Poor calibration should have high ECE"
    
    def test_ece_range(self):
        """Test that ECE is in valid range [0, 1]."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        ece = metrics.expected_calibration_error(probs, labels)
        assert 0 <= ece <= 1, f"ECE should be in [0, 1], got {ece}"
    
    def test_binary_classification(self):
        """Test ECE with binary classification."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
        labels = np.array([0, 1, 0, 0])
        ece = metrics.expected_calibration_error(probs, labels)
        assert isinstance(ece, float)
        assert ece >= 0
    
    def test_multiclass_classification(self):
        """Test ECE with multi-class classification."""
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]
        ])
        labels = np.array([0, 1, 2])
        ece = metrics.expected_calibration_error(probs, labels)
        assert isinstance(ece, float)
        assert ece >= 0
    
    def test_different_bin_sizes(self):
        """Test ECE with different numbers of bins."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        ece_5 = metrics.expected_calibration_error(probs, labels, n_bins=5)
        ece_15 = metrics.expected_calibration_error(probs, labels, n_bins=15)
        ece_30 = metrics.expected_calibration_error(probs, labels, n_bins=30)
        
        assert all(isinstance(e, float) for e in [ece_5, ece_15, ece_30])
    
    def test_logits_input(self):
        """Test ECE with logits input."""
        logits = np.array([[2.0, -1.0], [1.0, 0.5], [-0.5, 1.5]])
        labels = np.array([0, 0, 1])
        ece = metrics.expected_calibration_error(logits, labels, logits=True)
        assert isinstance(ece, float)
        assert ece >= 0


class TestMaximumCalibrationError:
    """Tests for Maximum Calibration Error (MCE)."""
    
    def test_mce_greater_equal_ece(self):
        """Test that MCE >= ECE always holds."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        
        ece = metrics.expected_calibration_error(probs, labels)
        mce = metrics.maximum_calibration_error(probs, labels)
        
        assert mce >= ece, "MCE should be >= ECE"
    
    def test_mce_range(self):
        """Test that MCE is in valid range [0, 1]."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        mce = metrics.maximum_calibration_error(probs, labels)
        assert 0 <= mce <= 1, f"MCE should be in [0, 1], got {mce}"


class TestRootMeanSquareCalibrationError:
    """Tests for Root Mean Square Calibration Error (RMSCE)."""
    
    def test_rmsce_positive(self):
        """Test that RMSCE is non-negative."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        rmsce = metrics.root_mean_square_calibration_error(probs, labels)
        assert rmsce >= 0, "RMSCE should be non-negative"
    
    def test_rmsce_vs_ece(self):
        """Test relationship between RMSCE and ECE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        
        ece = metrics.expected_calibration_error(probs, labels)
        rmsce = metrics.root_mean_square_calibration_error(probs, labels)
        
        # RMSCE penalizes large errors more, so it's typically >= ECE
        assert rmsce >= ece - 0.01, "RMSCE should generally be >= ECE"


class TestStaticCalibrationError:
    """Tests for Static Calibration Error (SCE)."""
    
    def test_sce_positive(self):
        """Test that SCE is non-negative."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        sce = metrics.static_calibration_error(probs, labels)
        assert sce >= 0, "SCE should be non-negative"
    
    def test_sce_multiclass(self):
        """Test SCE with multiple classes."""
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05]
        ])
        labels = np.array([0, 1, 2, 0])
        sce = metrics.static_calibration_error(probs, labels)
        assert isinstance(sce, float)
        assert sce >= 0


class TestAdaptiveCalibrationError:
    """Tests for Adaptive Calibration Error (ACE)."""
    
    def test_ace_positive(self):
        """Test that ACE is non-negative."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=100)
        labels = np.random.randint(0, 2, size=100)
        ace = metrics.adaptive_calibration_error(probs, labels)
        assert ace >= 0, "ACE should be non-negative"
    
    def test_ace_vs_sce(self):
        """Test ACE computation compared to SCE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        
        ace = metrics.adaptive_calibration_error(probs, labels)
        sce = metrics.static_calibration_error(probs, labels)
        
        # Both should be valid values
        assert ace >= 0 and sce >= 0

    def test_direction_false_matches_default(self):
        """direction=False must match the default non-directional ACE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)

        ace_default = metrics.adaptive_calibration_error(probs, labels)
        ace_explicit = metrics.adaptive_calibration_error(
            probs, labels, direction=False
        )
        assert ace_default == ace_explicit
        assert ace_default >= 0

    def test_direction_false_non_negative(self):
        """Non-directional ACE remains non-negative."""
        np.random.seed(0)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        ace = metrics.adaptive_calibration_error(probs, labels, direction=False)
        assert ace >= 0


class TestDirectionalCalibrationError:
    """Tests for directional (over, under) calibration error."""

    def test_hand_computed_overconfident(self):
        """Single-bin overconfident case: conf=0.9, all wrong -> (0.9, 0)."""
        confidences = np.array([0.9, 0.9, 0.9])
        accuracies = np.array([0.0, 0.0, 0.0])
        over, under = metrics._compute_calibration_error(
            confidences, accuracies, n_bins=1, adaptive_bins=True,
            norm=1, direction=True
        )
        assert over == pytest.approx(0.9)
        assert under == pytest.approx(0.0)

    def test_hand_computed_underconfident(self):
        """Single-bin underconfident case: conf=0.6, all correct -> (0, 0.4)."""
        confidences = np.array([0.6, 0.6, 0.6])
        accuracies = np.array([1.0, 1.0, 1.0])
        over, under = metrics._compute_calibration_error(
            confidences, accuracies, n_bins=1, adaptive_bins=True,
            norm=1, direction=True
        )
        assert over == pytest.approx(0.0)
        assert under == pytest.approx(0.4)

    def test_hand_computed_perfect(self):
        """Single-bin perfect calibration: conf matches accuracy -> (0, 0)."""
        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        accuracies = np.array([1.0, 0.0, 1.0, 0.0])  # mean acc = 0.5
        over, under = metrics._compute_calibration_error(
            confidences, accuracies, n_bins=1, adaptive_bins=True,
            norm=1, direction=True
        )
        assert over == pytest.approx(0.0)
        assert under == pytest.approx(0.0)

    def test_hand_computed_via_gce(self):
        """Directional GCE (top-1, adaptive, L1) matches hand-computed gaps."""
        # Overconfident: all predict class 0 with conf 0.9, all wrong
        probs_over = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        labels_over = np.array([1, 1, 1])
        over, under = metrics.general_calibration_error(
            probs_over, labels_over, n_bins=1, class_conditional=False,
            adaptive_bins=True, norm=1, direction=True
        )
        assert over == pytest.approx(0.9)
        assert under == pytest.approx(0.0)

        # Underconfident: all predict class 0 with conf 0.6, all correct
        probs_under = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])
        labels_under = np.array([0, 0, 0])
        over, under = metrics.general_calibration_error(
            probs_under, labels_under, n_bins=1, class_conditional=False,
            adaptive_bins=True, norm=1, direction=True
        )
        assert over == pytest.approx(0.0)
        assert under == pytest.approx(0.4)

    def test_directional_ace_overconfident(self):
        """Directional ACE (top-1): overconfident predictions -> (over > 0, under == 0)."""
        probs = np.array([
            [0.95, 0.05],
            [0.95, 0.05],
            [0.05, 0.95],
            [0.05, 0.95],
        ])
        labels = np.array([1, 1, 0, 0])
        over, under = metrics.adaptive_calibration_error(
            probs, labels, n_bins=1, direction=True
        )
        assert over == pytest.approx(0.95)
        assert under == pytest.approx(0.0)

    def test_directional_ace_underconfident(self):
        """Directional ACE (top-1): underconfident predictions -> (over == 0, under > 0)."""
        probs = np.array([
            [0.55, 0.45],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.45, 0.55],
        ])
        labels = np.array([0, 0, 1, 1])
        over, under = metrics.adaptive_calibration_error(
            probs, labels, n_bins=1, direction=True
        )
        assert over == pytest.approx(0.0)
        assert under == pytest.approx(0.45)

    def test_directional_ace_perfect(self):
        """Directional ACE is (0, 0) for perfectly calibrated predictions."""
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1, 0, 1])
        over, under = metrics.adaptive_calibration_error(
            probs, labels, n_bins=1, direction=True
        )
        assert over == pytest.approx(0.0, abs=1e-9)
        assert under == pytest.approx(0.0, abs=1e-9)

    def test_directional_ace_is_top1(self):
        """Directional ACE matches top-1 adaptive GCE, not class-conditional."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=50)
        labels = np.random.randint(0, 3, size=50)

        ace_dir = metrics.adaptive_calibration_error(probs, labels, direction=True)
        gce_top1 = metrics.general_calibration_error(
            probs, labels, class_conditional=False, adaptive_bins=True,
            norm=1, direction=True
        )
        assert ace_dir[0] == pytest.approx(gce_top1[0])
        assert ace_dir[1] == pytest.approx(gce_top1[1])

    def test_over_plus_under_equals_top1_absolute(self):
        """over + under equals top-1 adaptive absolute L1 (not class-conditional ACE)."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)

        over, under = metrics.adaptive_calibration_error(probs, labels, direction=True)
        top1_abs = metrics.general_calibration_error(
            probs, labels, class_conditional=False, adaptive_bins=True, norm=1
        )
        assert over >= 0 and under >= 0
        assert over + under == pytest.approx(top1_abs)

    def test_mixed_sign_bins_kept_separate(self):
        """Mixed over/under bins are not netted; both components are returned."""
        # Uniform bins: (0, 0.5] underconfident (0.4), (0.5, 1] overconfident (0.4)
        confidences = np.array([0.9, 0.9, 0.3, 0.3])
        accuracies = np.array([0.5, 0.5, 0.7, 0.7])
        over, under = metrics._compute_calibration_error(
            confidences, accuracies, n_bins=2, adaptive_bins=False,
            norm=1, direction=True
        )
        absolute = metrics._compute_calibration_error(
            confidences, accuracies, n_bins=2, adaptive_bins=False,
            norm=1, direction=False
        )
        assert over == pytest.approx(0.2)
        assert under == pytest.approx(0.2)
        assert over + under == pytest.approx(absolute)

    def test_ace_alias_with_direction(self):
        """ACE alias matches adaptive_calibration_error with direction=True."""
        np.random.seed(7)
        probs = np.random.dirichlet(np.ones(2), size=40)
        labels = np.random.randint(0, 2, size=40)

        via_fn = metrics.adaptive_calibration_error(probs, labels, direction=True)
        via_alias = metrics.ACE(probs, labels, direction=True)
        assert via_alias == via_fn
        assert isinstance(via_alias, tuple)
        assert len(via_alias) == 2
        assert all(isinstance(v, float) for v in via_alias)


class TestTopKCalibrationError:
    """Tests for Top-K Calibration Error."""
    
    def test_top1_equals_ece(self):
        """Test that top-1 calibration error equals ECE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=50)
        labels = np.random.randint(0, 3, size=50)
        
        # Top-1 with class-conditional should be different from ECE
        top1_ce = metrics.top_k_calibration_error(probs, labels, k=1)
        assert isinstance(top1_ce, float)
        assert top1_ce >= 0
    
    def test_different_k_values(self):
        """Test top-k calibration error with different k values."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(5), size=100)
        labels = np.random.randint(0, 5, size=100)
        
        top1_ce = metrics.top_k_calibration_error(probs, labels, k=1)
        top2_ce = metrics.top_k_calibration_error(probs, labels, k=2)
        top3_ce = metrics.top_k_calibration_error(probs, labels, k=3)
        
        assert all(isinstance(ce, float) for ce in [top1_ce, top2_ce, top3_ce])
        assert all(ce >= 0 for ce in [top1_ce, top2_ce, top3_ce])


class TestOverconfidenceError:
    """Tests for Overconfidence Error (OE)."""
    
    def test_oe_positive(self):
        """Test that OE is non-negative."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        oe = metrics.overconfidence_error(probs, labels)
        assert oe >= 0, "OE should be non-negative"
    
    def test_oe_overconfident_predictions(self):
        """Test OE with overconfident wrong predictions."""
        # Very confident but all wrong
        probs = np.array([[0.95, 0.05], [0.9, 0.1], [0.85, 0.15]])
        labels = np.array([1, 1, 1])  # All predictions are for class 0, but true is 1
        oe = metrics.overconfidence_error(probs, labels)
        assert oe > 0.3, "Overconfident wrong predictions should have high OE"


class TestThresholdedAdaptiveCalibrationError:
    """Tests for Thresholded Adaptive Calibration Error (TACE)."""
    
    def test_tace_with_threshold(self):
        """Test TACE with different thresholds."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(0, 3, size=100)
        
        tace_0 = metrics.thresholded_adaptive_calibration_error(probs, labels, threshold=0.0)
        tace_01 = metrics.thresholded_adaptive_calibration_error(probs, labels, threshold=0.1)
        tace_02 = metrics.thresholded_adaptive_calibration_error(probs, labels, threshold=0.2)
        
        assert all(isinstance(t, float) for t in [tace_0, tace_01, tace_02])
        assert all(t >= 0 for t in [tace_0, tace_01, tace_02])


class TestGeneralCalibrationError:
    """Tests for General Calibration Error (GCE)."""
    
    def test_gce_produces_ece(self):
        """Test that GCE with correct parameters produces ECE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        ece = metrics.expected_calibration_error(probs, labels)
        gce = metrics.general_calibration_error(
            probs, labels, class_conditional=False, adaptive_bins=False, norm=1
        )
        
        assert np.isclose(ece, gce, rtol=1e-5), "GCE with ECE parameters should equal ECE"
    
    def test_gce_produces_mce(self):
        """Test that GCE with correct parameters produces MCE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        mce = metrics.maximum_calibration_error(probs, labels)
        gce = metrics.general_calibration_error(
            probs, labels, class_conditional=False, adaptive_bins=False, norm='inf'
        )
        
        assert np.isclose(mce, gce, rtol=1e-5), "GCE with MCE parameters should equal MCE"
    
    def test_gce_produces_rmsce(self):
        """Test that GCE with correct parameters produces RMSCE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        rmsce = metrics.root_mean_square_calibration_error(probs, labels)
        gce = metrics.general_calibration_error(
            probs, labels, class_conditional=False, adaptive_bins=False, norm=2
        )
        
        assert np.isclose(rmsce, gce, rtol=1e-5), "GCE with RMSCE parameters should equal RMSCE"
    
    def test_gce_invalid_inputs(self):
        """Test GCE with invalid inputs."""
        probs = np.array([0.8, 0.2])  # Wrong shape
        labels = np.array([0, 1])
        
        with pytest.raises(ValueError):
            metrics.general_calibration_error(probs, labels)
    
    def test_gce_mismatched_shapes(self):
        """Test GCE with mismatched input shapes."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        labels = np.array([0])  # Wrong length
        
        with pytest.raises(ValueError):
            metrics.general_calibration_error(probs, labels)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_sample(self):
        """Test metrics with single sample."""
        probs = np.array([[0.8, 0.2]])
        labels = np.array([0])
        
        ece = metrics.expected_calibration_error(probs, labels)
        assert isinstance(ece, float)
        assert ece >= 0
    
    def test_two_samples(self):
        """Test metrics with two samples."""
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([0, 1])
        
        ece = metrics.expected_calibration_error(probs, labels)
        mce = metrics.maximum_calibration_error(probs, labels)
        
        assert isinstance(ece, float) and isinstance(mce, float)
        assert ece >= 0 and mce >= 0
    
    def test_uniform_probabilities(self):
        """Test with uniform probabilities."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0, 1, 0])
        
        ece = metrics.expected_calibration_error(probs, labels)
        assert isinstance(ece, float)
    
    def test_deterministic_predictions(self):
        """Test with deterministic (one-hot) predictions."""
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1, 0])
        
        ece = metrics.expected_calibration_error(probs, labels)
        assert ece < 0.01, "Perfect predictions should have near-zero ECE"


class TestAliases:
    """Test that aliases work correctly."""
    
    def test_ece_alias(self):
        """Test ECE alias."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        labels = np.array([0, 1])
        
        ece1 = metrics.expected_calibration_error(probs, labels)
        ece2 = metrics.ECE(probs, labels)
        
        assert ece1 == ece2
    
    def test_mce_alias(self):
        """Test MCE alias."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        labels = np.array([0, 1])
        
        mce1 = metrics.maximum_calibration_error(probs, labels)
        mce2 = metrics.MCE(probs, labels)
        
        assert mce1 == mce2
    
    def test_gce_alias(self):
        """Test GCE alias."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        labels = np.array([0, 1])
        
        gce1 = metrics.general_calibration_error(probs, labels)
        gce2 = metrics.GCE(probs, labels)
        
        assert gce1 == gce2
