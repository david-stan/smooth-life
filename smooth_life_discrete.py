import numpy as np
from scipy.special import j1
from scipy.fft import fft2, ifft2
from matplotlib import pyplot as plt
from matplotlib import animation

class FourierWeights:
    def __init__(self, res, h=7, eps=1e-10):
        self.res = res              # Grid resolution
        self.h = h                  # Radius
        self.eps = eps

        self.k_radial = None
        self.disk_fft = None
        self.annulus_fft = None

        # Precompute radial frequencies and filters
        self._precompute_frequencies()
        self._precompute_disk()
        self._precompute_annulus()


    def _precompute_frequencies(self):
        # Compute frequencies in Fourier space (kx, ky)
        kx = np.fft.fftfreq(self.res, d=1.0)
        ky = np.fft.fftfreq(self.res, d=1.0)
        KX, KY = np.meshgrid(kx, ky)
        self.k_radial = np.sqrt(KX**2 + KY**2)

        self.k_radial[self.k_radial == 0] = self.eps

    def _precompute_disk(self):
        # Compute the Fourier transform of the disk indicator function
        # using the bessel functions of the first order
        self.disk_fft = (np.sqrt(3 * self.h) / (4 * self.k_radial)) * j1(2 * np.pi * self.h * self.k_radial)

    def _precompute_annulus(self):
        # Compute the Fourier transform for the annulus (difference between disks)
        disk_3h_fft = (np.sqrt(9 * self.h) / (4 * self.k_radial)) * j1(6 * np.pi * self.h * self.k_radial)
        self.annulus_fft = disk_3h_fft - self.disk_fft

    def compute_M(self, f_fft):
        # Multiply Fourier transform of f(x, y) by the Fourier transform of the disk
        M_fft = f_fft * self.disk_fft
        # M_fft = self.apply_gaussian_filter(M_fft)
        # Compute the inverse Fourier transform to get M(x, y) in real space
        M = (1 / (2 * np.pi * self.h ** 2)) * ifft2(M_fft).real
        return M
    
    def compute_N(self, f_fft):
        # Multiply Fourier transform of f(x, y) by the Fourier transform of the annulus
        N_fft = f_fft * self.annulus_fft
        # N_fft = self.apply_gaussian_filter(N_fft)
        # Compute the inverse Fourier transform to get N(x, y) in real space
        N = (1 / (8 * np.pi * self.h ** 2)) * ifft2(N_fft).real
        return N

    def apply_gaussian_filter(self, field_fft, cutoff=0.66):
        kx = np.fft.fftfreq(self.res, d=1.0)
        ky = np.fft.fftfreq(self.res, d=1.0)
        KX, KY = np.meshgrid(kx, ky)
        omega = np.sqrt(KX**2 + KY**2)
        
        # Gaussian low-pass filter in the Fourier domain
        gaussian_filter = np.exp(-(omega**2) / (2 * cutoff**2))
        return field_fft * gaussian_filter

class Rules:
    # Birth span
    B0, B1 = 0.278, 0.365

    # Death span
    D0, D1 = 0.267, 0.445

    # Alpha
    alpha_M, alpha_N = 0.147, 0.028

    def logistic_threshold(self, x, x0, alpha):
        """Logistic function on x around x0 with transition width alpha

        Approximately:
            (x - alpha/2) < x0 : 0
            (x + alpha/2) > x0 : 1

        """
        return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - x0)))
    
    def logistic_interval(self, x, a, b, alpha):
        """Logistic function on x between a and b with transition width alpha

        Very approximately:
            x < a     : 0
            a < x < b : 1
            x > b     : 0

        """
        return self.logistic_threshold(x, a, alpha) * (1.0 - self.logistic_threshold(x, b, alpha))
    
    def lerp(self, a, b, t):
        """Linear intererpolate from a to b with t ranging [0,1]

        """
        return (1.0 - t) * a + t * b


    def S(self, M, N):
        """State transition function
        """
        # Convert the local cell average `m` to a metric of how alive the local cell is.
        # We transition around 0.5 (0 is fully dead and 1 is fully alive).
        # The transition width is set by `self.M`
        aliveness = self.logistic_threshold(M, 0.5, self.alpha_M)
        # A fully dead cell will become alive if the neighbor density is between B1 and B2.
        # A fully alive cell will stay alive if the neighhbor density is between D1 and D2.
        # Interpolate between the two sets of thresholds depending on how alive/dead the cell is.
        threshold1 = self.lerp(self.B0, self.D0, aliveness)
        threshold2 = self.lerp(self.B1, self.D1, aliveness)
        # Now with the smoothness of `logistic_interval` determine if the neighbor density is
        # inside of the threshold to stay/become alive.
        new_aliveness = self.logistic_interval(N, threshold1, threshold2, self.alpha_N)

        return np.clip(new_aliveness, 0, 1)


class SmoothLife:
    def __init__(self):
        self.weights = FourierWeights(1 << 8)
        self.rules = Rules()

        self.field = np.zeros((self.weights.res, self.weights.res))
        self.initialize_field(self.weights.res, self.weights.h)
        self.field_fft = fft2(self.field)

    def initialize_field(self, res, h):
        """Populate field with random living squares

        If count unspecified, do a moderately dense fill
        """
        count = int(res**2 / ((h * 3 * 2) ** 2))
        for _ in range(count):
            radius = int(3 * h)
            r = np.random.randint(0, res - radius)
            c = np.random.randint(0, res - radius)
            self.field[r : r + radius, c : c + radius] = 1
        
    def step(self):
        M_buffer = self.weights.compute_M(self.field_fft)
        N_buffer = self.weights.compute_N(self.field_fft)
        self.field = self.rules.S(M_buffer, N_buffer)
        self.field_fft = fft2(self.field)
        return self.field


def show_animation():
    sl = SmoothLife()
    sl.step()

    fig = plt.figure()
    # Nice color maps: viridis, plasma, gray, binary, seismic, gnuplot
    im = plt.imshow(
        sl.field, animated=True, cmap=plt.get_cmap("viridis"), aspect="equal"
    )

    def animate(*args):
        im.set_array(sl.step())
        return (im,)

    ani = animation.FuncAnimation(fig, animate, interval=60, blit=True)
    plt.show()

def main():
    show_animation()

if __name__ == "__main__":
    main()
