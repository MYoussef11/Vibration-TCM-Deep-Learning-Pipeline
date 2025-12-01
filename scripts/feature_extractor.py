"""
Simplified Feature Extractor for Top 20 ML Features.
Optimized for real-time streaming with minimal computation.
"""

import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt


class SimplifiedFeatureExtractor:
    """Extract top 20 features for ML model inference."""
    
    def __init__(self, sampling_rate=10):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 10)
        """
        self.sampling_rate = sampling_rate
        
        # Top 20 feature names (from model)
        self.feature_names = [
            'acceleration_yg_wpd_wpd_energy_lvl3_node0',
            'acceleration_yg_freq_spectral_energy',
            'acceleration_yg_freq_band_power_low',
            'acceleration_yg_time_rms',
            'acceleration_zg_freq_spectral_energy',
            'acceleration_zg_time_rms',
            'acceleration_yg_time_mean',
            'acceleration_zg_freq_band_power_low',
            'acceleration_zg_wpd_wpd_energy_lvl3_node0',
            'acceleration_zg_time_crest_factor',
            'acceleration_zg_time_mean',
            'angular_velocity_ydps_time_mad',
            'acceleration_zg_time_skewness',
            'acceleration_zg_time_std',
            'angular_velocity_zdps_wpd_wpd_energy_lvl3_node4',
            'angular_velocity_ydps_time_rms',
            'angular_velocity_ydps_time_std',
            'acceleration_zg_time_mad',
            'acceleration_zg_wpd_wpd_energy_lvl3_node1',
            'angular_velocity_ydps_wpd_wpd_energy_lvl3_node4'
        ]
        
    def extract(self, window_data):
        """
        Extract top 20 features from a window of sensor data.
        
        Args:
            window_data: numpy array of shape (n_samples, 6)
                        [ax, ay, az, wx, wy, wz]
        
        Returns:
            dict: Feature name -> value
        """
        features = {}
        
        # Extract channels
        ax, ay, az = window_data[:, 0], window_data[:, 1], window_data[:, 2]
        wx, wy, wz = window_data[:, 3], window_data[:, 4], window_data[:, 5]
        
        # === Y-axis Acceleration Features ===
        features['acceleration_yg_time_rms'] = np.sqrt(np.mean(ay**2))
        features['acceleration_yg_time_mean'] = np.mean(ay)
        
        # Frequency domain for Y
        freqs_y, psd_y = signal.periodogram(ay, fs=self.sampling_rate)
        features['acceleration_yg_freq_spectral_energy'] = np.sum(psd_y)
        features['acceleration_yg_freq_band_power_low'] = np.sum(psd_y[freqs_y < 2.5])
        
        # Wavelet for Y
        coeffs_y = pywt.wavedec(ay, 'db4', level=3)
        features['acceleration_yg_wpd_wpd_energy_lvl3_node0'] = np.sum(coeffs_y[0]**2)
        
        # === Z-axis Acceleration Features ===
        features['acceleration_zg_time_rms'] = np.sqrt(np.mean(az**2))
        features['acceleration_zg_time_mean'] = np.mean(az)
        features['acceleration_zg_time_std'] = np.std(az)
        features['acceleration_zg_time_mad'] = np.median(np.abs(az - np.median(az)))
        features['acceleration_zg_time_skewness'] = skew(az)
        features['acceleration_zg_time_crest_factor'] = np.max(np.abs(az)) / features['acceleration_zg_time_rms'] if features['acceleration_zg_time_rms'] > 0 else 0
        
        # Frequency domain for Z
        freqs_z, psd_z = signal.periodogram(az, fs=self.sampling_rate)
        features['acceleration_zg_freq_spectral_energy'] = np.sum(psd_z)
        features['acceleration_zg_freq_band_power_low'] = np.sum(psd_z[freqs_z < 2.5])
        
        # Wavelet for Z
        coeffs_z = pywt.wavedec(az, 'db4', level=3)
        features['acceleration_zg_wpd_wpd_energy_lvl3_node0'] = np.sum(coeffs_z[0]**2)
        features['acceleration_zg_wpd_wpd_energy_lvl3_node1'] = np.sum(coeffs_z[1]**2) if len(coeffs_z) > 1 else 0
        
        # === Y-axis Angular Velocity Features ===
        features['angular_velocity_ydps_time_rms'] = np.sqrt(np.mean(wy**2))
        features['angular_velocity_ydps_time_std'] = np.std(wy)
        features['angular_velocity_ydps_time_mad'] = np.median(np.abs(wy - np.median(wy)))
        
        # Wavelet for Y angular velocity
        coeffs_wy = pywt.wavedec(wy, 'db4', level=3)
        features['angular_velocity_ydps_wpd_wpd_energy_lvl3_node4'] = np.sum(coeffs_wy[-1]**2) if len(coeffs_wy) > 4 else 0
        
        # === Z-axis Angular Velocity Features ===
        coeffs_wz = pywt.wavedec(wz, 'db4', level=3)
        features['angular_velocity_zdps_wpd_wpd_energy_lvl3_node4'] = np.sum(coeffs_wz[-1]**2) if len(coeffs_wz) > 4 else 0
        
        return features
    
    def extract_array(self, window_data):
        """
        Extract features and return as ordered array.
        
        Args:
            window_data: numpy array of shape (n_samples, 6)
        
        Returns:
            numpy array of shape (20,)
        """
        features_dict = self.extract(window_data)
        return np.array([features_dict[name] for name in self.feature_names])


# Test function
if __name__ == "__main__":
    import time
    
    # Create dummy data
    np.random.seed(42)
    test_window = np.random.randn(20, 6) * 0.1
    test_window[:, 2] += 9.8  # Add gravity to Z
    
    extractor = SimplifiedFeatureExtractor()
    
    # Time the extraction
    start = time.time()
    for _ in range(100):
        features = extractor.extract_array(test_window)
    elapsed = (time.time() - start) / 100 * 1000
    
    print(f"✅ Feature extraction time: {elapsed:.2f}ms per window")
    print(f"✅ Extracted {len(features)} features")
    print(f"\nSample features:")
    feat_dict = extractor.extract(test_window)
    for i, (name, val) in enumerate(list(feat_dict.items())[:5]):
        print(f"  {i+1}. {name}: {val:.6f}")
