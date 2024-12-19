import numpy as np

from core.transformer.standard_scaler import StandardScaler


def test_standard_scaler_fit():
    X = range(10)
    scaler = StandardScaler()
    scaler.fit(X)
    np.testing.assert_almost_equal(scaler.mean_, np.mean(X))
    np.testing.assert_almost_equal(scaler.scale_, np.std(X))


def test_standard_scaler_transform():
    X = range(10)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    expected_scaled = (X - np.mean(X)) / np.std(X)
    np.testing.assert_almost_equal(X_scaled, expected_scaled)


def test_standard_scaler_fit_transform():
    X = range(10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    expected_scaled = (X - np.mean(X)) / np.std(X)
    np.testing.assert_almost_equal(X_scaled, expected_scaled)


def test_standard_scaler_with_different_data():
    X = np.random.normal(2, 3, 1000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    np.testing.assert_almost_equal(np.mean(X_scaled, axis=0), 0, decimal=6)
    np.testing.assert_almost_equal(np.std(X_scaled, axis=0), 1, decimal=6)
