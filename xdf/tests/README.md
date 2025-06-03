# Unit Tests for `autocorr_pearson_scrub` Function

This document describes the comprehensive pytest-based unit test suite for the `autocorr_pearson_scrub` function in the xDF package.

## Overview

The `autocorr_pearson_scrub` function calculates correlation and variance matrices that account for autocorrelation in time series data, with support for censored/scrubbed volumes (NaN values). This test suite ensures the function works correctly across various scenarios and edge cases.

## Test Structure

The tests are organized in the `TestAutocorrPearsonScrub` class and cover:

### 1. Basic Functionality Tests

- **`test_basic_functionality_no_scrub`**: Tests basic operation without scrubbed frames
- **`test_with_scrubbed_frames`**: Tests functionality with NaN values (scrubbed volumes)
- **`test_different_methods`**: Tests different regularization methods (truncate adaptive/fixed)

### 2. Statistical Property Tests

- **`test_autocorrelated_data_variance_increase`**: Verifies that autocorrelated data produces larger variance estimates
- **`test_correlation_bounds`**: Ensures correlation coefficients are within [-1, 1]
- **`test_variance_positivity`**: Checks that variance estimates are non-negative
- **`test_statistical_properties`**: Tests overall statistical behavior with known correlations
- **`test_known_autocorrelation_scenario`**: Tests a specific scenario with controlled autocorrelation

### 3. Input/Output Validation Tests

- **`test_input_validation`**: Tests error handling for invalid inputs
- **`test_symmetry_properties`**: Verifies that output matrices are symmetric
- **`test_limit_variance_parameter`**: Tests the variance limiting functionality
- **`test_copy_parameter`**: Ensures the copy parameter works correctly

### 4. Edge Case Tests

- **`test_single_feature_edge_case`**: Tests behavior with single time series
- **`test_heavy_scrubbing`**: Tests with many scrubbed frames (30% of data)
- **`test_reproducibility`**: Ensures consistent results with same input

## Key Test Features

### Synthetic Data Generation

The test suite includes helper methods to generate realistic test data:

- **`generate_autocorrelated_timeseries`**: Creates AR(1) time series with specified autocorrelation
- **`generate_correlated_timeseries`**: Creates two correlated autocorrelated time series

### Expected Outputs

The tests verify that the function returns a dictionary with these keys:
- `"r"`: Correlation coefficient matrix
- `"p"`: P-value matrix
- `"z"`: Z-scores adjusted for autocorrelation
- `"z_uncorrected"`: Z-scores without autocorrelation correction
- `"v"`: Variance of correlation coefficients
- `"var_z"`: Variance of z-transformed correlation coefficients
- `"varlimit"`: Theoretical variance limit
- `"varlimit_idx"`: Indices where variance exceeded theoretical limit

### Statistical Expectations

The tests verify several important statistical properties:

1. **Autocorrelation Effect**: Autocorrelated data should generally produce larger variance estimates than white noise
2. **Correlation Bounds**: All correlations should be between -1 and 1
3. **Matrix Symmetry**: Correlation and variance matrices should be symmetric
4. **Diagonal Properties**:
   - Correlation matrix diagonal should be 1 (self-correlation)
   - Variance matrix diagonal should be 0
5. **Variance Positivity**: All variance estimates should be non-negative

## Running the Tests

To run all tests:
```bash
pytest test_autocorr_pearson_scrub.py -v
```

To run a specific test:
```bash
pytest test_autocorr_pearson_scrub.py::TestAutocorrPearsonScrub::test_basic_functionality_no_scrub -v
```

## Test Parameters

The tests use various parameter combinations to ensure robustness:

- **Sample sizes**: 50-200 time points
- **Feature counts**: 1-4 time series
- **Autocorrelation levels**: 0.3-0.7
- **Scrubbing levels**: 0-30% of frames
- **Methods**: "truncate" with "adaptive" or fixed integer parameters

## Known Limitations

1. **Tukey Method**: Currently commented out due to dimension compatibility issues in the implementation
2. **Statistical Variance**: Some tests use tolerances to account for statistical variation in synthetic data
3. **Diagonal Elements**: Z-score matrices have infinite diagonal elements due to perfect self-correlation (expected behavior)

## Expected Warnings

The tests may produce these expected warnings:
- `RuntimeWarning: divide by zero encountered in arctanh`: Due to perfect correlations (diagonal elements)
- `RuntimeWarning: invalid value encountered in divide`: Related to division by zero in variance calculations

These warnings are expected and do not indicate test failures.

## Test Validation

The test suite validates that:
1. The function handles various input formats correctly
2. Statistical properties are maintained under autocorrelation
3. Scrubbing/censoring of volumes works properly
4. Edge cases are handled gracefully
5. Results are reproducible and mathematically consistent

This comprehensive test suite ensures the `autocorr_pearson_scrub` function performs correctly across a wide range of realistic neuroimaging data scenarios.