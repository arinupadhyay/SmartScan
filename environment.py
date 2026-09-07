# environment.py
import numpy as np

# ==========================================
# 1. EMITTER CLASSES (COMPLEX SIGNAL BEHAVIORS)
# ==========================================

class Emitter:
    """Base class for tactical signal sources."""
    def __init__(self, emitter_type, num_bands, tx_power_dbm=30.0, seed=None):
        self.type = emitter_type
        self.num_bands = num_bands
        self.tx_power_dbm = tx_power_dbm
        self.rng = np.random.default_rng(seed)

    def generate_presence_grid(self, num_time_slots):
        raise NotImplementedError


class RadarChirpEmitter(Emitter):
    """
    Search/Track Radar: High duty cycle, periodic pulse sequence.
    """
    def __init__(self, num_bands, dwell_time=12, duty_cycle=0.4, target_bands=None, tx_power_dbm=45.0, seed=None):
        super().__init__("Radar_Chirp", num_bands, tx_power_dbm, seed)
        self.dwell_time = dwell_time
        self.duty_cycle = duty_cycle
        self.target_bands = target_bands if target_bands is not None else [0, num_bands // 2]

    def generate_presence_grid(self, num_time_slots):
        grid = np.zeros((num_time_slots, self.num_bands), dtype=float)
        pulse_width = max(1, int(self.dwell_time * self.duty_cycle))
        
        for t in range(num_time_slots):
            if (t % self.dwell_time) < pulse_width:
                for b in self.target_bands:
                    if b < self.num_bands:
                        grid[t, b] = 1.0
        return grid


class FHSSEmitter(Emitter):
    """
    Frequency Hopping Spread Spectrum (FHSS): Rapid pseudo-random band switching.
    """
    def __init__(self, num_bands, dwell_time=3, tx_power_dbm=30.0, seed=None):
        super().__init__("FHSS", num_bands, tx_power_dbm, seed)
        self.dwell_time = max(1, dwell_time)

    def generate_presence_grid(self, num_time_slots):
        grid = np.zeros((num_time_slots, self.num_bands), dtype=float)
        current_band = self.rng.integers(0, self.num_bands)
        
        for t in range(num_time_slots):
            if t > 0 and t % self.dwell_time == 0:
                available = [b for b in range(self.num_bands) if b != current_band]
                current_band = self.rng.choice(available)
            grid[t, current_band] = 1.0
        return grid


class BurstCommsEmitter(Emitter):
    """
    Tactical Burst Comms: Low duty cycle, short duration, unpredictable bursts.
    """
    def __init__(self, num_bands, burst_prob=0.12, max_pulse_width=3, tx_power_dbm=25.0, seed=None):
        super().__init__("Burst_Comms", num_bands, tx_power_dbm, seed)
        self.burst_prob = burst_prob
        self.max_pulse_width = max_pulse_width

    def generate_presence_grid(self, num_time_slots):
        grid = np.zeros((num_time_slots, self.num_bands), dtype=float)
        t = 0
        while t < num_time_slots:
            if self.rng.random() < self.burst_prob:
                target_band = self.rng.integers(0, self.num_bands)
                pw = self.rng.integers(1, self.max_pulse_width + 1)
                grid[t : min(t + pw, num_time_slots), target_band] = 1.0
                t += pw
            else:
                t += 1
        return grid


# ==========================================
# 2. ADVERSARY / ACTIVE JAMMER CLASS
# ==========================================

class AdversaryJammer:
    """
    Adversarial Electronic Countermeasures (ECM) Agent.
    """
    def __init__(self, num_bands, strategy="SWEEP", jammer_power_dbm=50.0, seed=None):
        self.num_bands = num_bands
        self.strategy = strategy.upper()
        self.jammer_power_dbm = jammer_power_dbm
        self.rng = np.random.default_rng(seed)

    def generate_jamming_grid(self, num_time_slots, target_presence=None):
        grid = np.zeros((num_time_slots, self.num_bands), dtype=float)
        
        if self.strategy == "BARRAGE":
            jammed_channels = self.rng.choice(self.num_bands, size=max(1, self.num_bands // 2), replace=False)
            grid[:, jammed_channels] = 1.0
            
        elif self.strategy == "SWEEP":
            for t in range(num_time_slots):
                active_band = (t // 2) % self.num_bands
                grid[t, active_band] = 1.0
                
        elif self.strategy == "REACTIVE_SPOOF":
            if target_presence is not None:
                for t in range(num_time_slots):
                    active_targets = np.where(target_presence[t] > 0)[0]
                    if len(active_targets) > 0:
                        spoof_band = (self.rng.choice(active_targets) + 1) % self.num_bands
                        grid[t, spoof_band] = 1.0
        return grid


# ==========================================
# 3. RF PHYSICAL ENVIRONMENT (SNR, FADING, NOISE)
# ==========================================

class RFEnvironment:
    """
    Tactical Spectrum Environment combining Emitters, Jamming, Path Loss,
    Rayleigh Fading, and AWGN.
    """
    def __init__(
        self, 
        num_bands=10, 
        num_time_slots=200, 
        noise_floor_dbm=-100.0, 
        detection_threshold_dbm=-85.0,
        enable_fading=True,
        jammer_strategy="SWEEP",
        seed=42
    ):
        self.num_bands = num_bands
        self.num_time_slots = num_time_slots
        self.noise_floor_dbm = noise_floor_dbm
        self.detection_threshold_dbm = detection_threshold_dbm
        self.enable_fading = enable_fading
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.emitter_distance_km = 15.0
        self.path_loss_exponent = 3.2

        self.signal_grid, self.jamming_grid, self.power_grid_dbm, self.grid = self._synthesize_rf_spectrum(jammer_strategy)

    def _dbm_to_linear(self, dbm):
        return 10.0 ** (dbm / 10.0)

    def _linear_to_dbm(self, linear_mw):
        return 10.0 * np.log10(np.maximum(linear_mw, 1e-15))

    def _calculate_path_loss(self, distance_km, freq_mhz=2400.0):
        return 32.44 + 20 * np.log10(freq_mhz) + 10 * self.path_loss_exponent * np.log10(distance_km)

    def _synthesize_rf_spectrum(self, jammer_strategy):
        radar = RadarChirpEmitter(self.num_bands, dwell_time=10, duty_cycle=0.3, target_bands=[1, self.num_bands-2], tx_power_dbm=45.0, seed=self.seed)
        fhss = FHSSEmitter(self.num_bands, dwell_time=4, tx_power_dbm=30.0, seed=self.seed + 1)
        burst = BurstCommsEmitter(self.num_bands, burst_prob=0.10, max_pulse_width=3, tx_power_dbm=25.0, seed=self.seed + 2)

        p_radar = radar.generate_presence_grid(self.num_time_slots)
        p_fhss = fhss.generate_presence_grid(self.num_time_slots)
        p_burst = burst.generate_presence_grid(self.num_time_slots)

        signal_presence = np.maximum(p_radar, np.maximum(p_fhss, p_burst))

        jammer = AdversaryJammer(self.num_bands, strategy=jammer_strategy, jammer_power_dbm=50.0, seed=self.seed + 3)
        jamming_presence = jammer.generate_jamming_grid(self.num_time_slots, signal_presence)

        path_loss_dB = self._calculate_path_loss(self.emitter_distance_km)
        power_grid_dbm = np.ones((self.num_time_slots, self.num_bands)) * self.noise_floor_dbm

        for t in range(self.num_time_slots):
            for b in range(self.num_bands):
                sig_pwr_mw = 0.0
                if p_radar[t, b]:
                    sig_pwr_mw += self._dbm_to_linear(radar.tx_power_dbm - path_loss_dB)
                if p_fhss[t, b]:
                    sig_pwr_mw += self._dbm_to_linear(fhss.tx_power_dbm - path_loss_dB)
                if p_burst[t, b]:
                    sig_pwr_mw += self._dbm_to_linear(burst.tx_power_dbm - path_loss_dB)

                jam_pwr_mw = 0.0
                if jamming_presence[t, b]:
                    jam_pwr_mw += self._dbm_to_linear(jammer.jammer_power_dbm - (path_loss_dB * 0.6))

                if self.enable_fading and (sig_pwr_mw > 0 or jam_pwr_mw > 0):
                    fading_factor = self.rng.rayleigh(scale=1.0) ** 2
                    sig_pwr_mw *= fading_factor
                    jam_pwr_mw *= fading_factor

                noise_mw = self._dbm_to_linear(self.noise_floor_dbm)
                awgn_variance = self._dbm_to_linear(self.noise_floor_dbm + 3.0)
                sampled_noise_mw = max(1e-15, self.rng.normal(noise_mw, awgn_variance * 0.1))

                total_pwr_mw = sig_pwr_mw + jam_pwr_mw + sampled_noise_mw
                power_grid_dbm[t, b] = self._linear_to_dbm(total_pwr_mw)

        detected_binary_grid = (power_grid_dbm >= self.detection_threshold_dbm).astype(int)

        return signal_presence, jamming_presence, power_grid_dbm, detected_binary_grid

    def get_slot_activity(self, t):
        if 0 <= t < self.num_time_slots:
            return self.grid[t, :]
        return np.zeros(self.num_bands, dtype=int)

    def get_slot_power(self, t):
        if 0 <= t < self.num_time_slots:
            return self.power_grid_dbm[t, :]
        return np.ones(self.num_bands) * self.noise_floor_dbm