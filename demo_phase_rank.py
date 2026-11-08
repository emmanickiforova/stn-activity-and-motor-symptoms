"""Расчёт фазовой привязки на одной синтетической записи."""
import numpy as np
from scipy import signal


def phase_vectors(analytic, spike_times, fs):
    """Интерполяция комплексного сигнала в моменты спайков."""
    positions = spike_times * fs
    left = np.clip(np.floor(positions).astype(int), 0, len(analytic) - 2)
    weight = np.clip(positions - left, 0, 1)
    vectors = analytic[left] * (1 - weight) + analytic[left + 1] * weight
    return vectors / np.abs(vectors)


def ppc(vectors):
    n = len(vectors)
    return (np.abs(vectors.sum()) ** 2 - n) / (n * (n - 1))


def phase_rank(lfp, spike_times, fs, rng):
    band = signal.butter(3, [13, 30], btype="band", fs=fs, output="sos")
    analytic = signal.hilbert(signal.sosfiltfilt(band, lfp))
    duration = len(lfp) / fs
    observed = ppc(phase_vectors(analytic, spike_times, fs))
    null = []
    for shift in rng.uniform(2, duration - 2, size=500):
        shifted_times = (spike_times + shift) % duration
        null.append(ppc(phase_vectors(analytic, shifted_times, fs)))
    null = np.asarray(null)
    rank = (np.sum(null < observed) + 0.5 * np.sum(null == observed) + 0.5) / (len(null) + 1)
    return observed, rank


def beta_over_background(lfp, fs):
    """Среднее превышение log10 мощности бета-диапазона над лог-линейным фоном."""
    freq, power = signal.welch(lfp, fs, nperseg=int(2 * fs))
    beta = (freq >= 13) & (freq <= 30)
    mains = (freq >= 45) & (freq <= 55)
    fit = (freq >= 4) & (freq <= 80) & ~beta & ~mains
    log_power = np.log10(power + 1e-30)
    coeff = np.polyfit(np.log10(freq[fit]), log_power[fit], 1)
    background = np.polyval(coeff, np.log10(freq[beta]))
    return np.mean(log_power[beta] - background)


def synthetic_recording():
    rng = np.random.default_rng(42)
    fs = 500
    time = np.arange(60 * fs) / fs
    frequency = 20 + 3 * np.sin(2 * np.pi * 0.13 * time)
    phase = 2 * np.pi * np.cumsum(frequency) / fs
    lfp = np.cos(phase) + rng.normal(0, 0.5, size=len(time))
    # В демонстрации намеренно повышаем вероятность спайка около гребня.
    spike_probability = (12 / fs) * (1 + 0.7 * np.cos(phase))
    spike_times = time[rng.random(len(time)) < spike_probability]
    return lfp, spike_times, fs


def main():
    lfp, spike_times, fs = synthetic_recording()
    observed, rank = phase_rank(lfp, spike_times, fs, np.random.default_rng(10))
    print(f"Бета над фоном = {beta_over_background(lfp, fs):.3f} (логарифмическая шкала)")
    print(f"Синтетический пример: {len(spike_times)} спайков, PPC = {observed:.4f}, ранг = {rank:.3f}")


if __name__ == "__main__":
    main()
