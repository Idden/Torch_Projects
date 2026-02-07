import numpy as np
from IPython.display import display, Math

# finds the closest int larger than input that is a power of 2
def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() # bit shift

# input int and output np.ndarray
def haar_matrix_builder(n: int) -> np.ndarray:

    # base case and dim check
    assert n > 0 and (n & (n - 1)) == 0, "n must be power of 2"
    if n == 1:
        return np.array([[1.0]], dtype=float)

    half = n // 2
    #print(type(half))
    norm = 1.0 / np.sqrt(2.0)

    # build the a haar level
    B = np.zeros((n, n), dtype=float)
    for k in range(half):
        col1 = 2 * k
        col2 = 2 * k + 1
        B[k, col1] = norm
        B[k, col2] = norm
        B[half+k, col1] = norm
        B[half+k, col2] = -norm

    # recursive call to build haar levels
    W_half = haar_matrix_builder(half)

    # after hitting base case multiply all haar levels
    R = np.block([
        [W_half, np.zeros((half, half), dtype=float)],
        [np.zeros((half, half), dtype=float), np.eye(half, dtype=float)]
    ])

    # def matrix_to_latex(mat):
    #     rows = [" & ".join(map(str,row)) for row in mat]
    #     body = r"\\ ".join(rows)
    #     return r"\begin{bmatrix}" + body + r"\end{bmatrix}"

    # display(Math(matrix_to_latex(np.round(R @ B, 3))))
    return R @ B

# input np.ndarray and output np.ndarray
def haar_transform_2d_classical(x: np.ndarray, inverse=False) -> np.ndarray:
    # build the haar transform classically
    H, W = x.shape
    WH = haar_matrix_builder(H)
    WW = haar_matrix_builder(W)

    # inverse if true
    if inverse:
        WH = WH.T
        WW = WW.T

    #print(f"{WH}\n\n\n")

    # apply the haar transform
    return WH @ x @ WW.T

# apply X pauli matrix
def apply_x_to_statevector(sv: np.ndarray, n_qubits: int, q: int) -> np.ndarray:
    psi = sv.reshape([2] * n_qubits)
    psi = np.swapaxes(psi, q, 0)
    psi = psi[::-1, ...] # reverse order of amplitudes to apply X gate
    psi = np.swapaxes(psi, 0, q)
    return psi.reshape(-1)

# apply Z pauli matrix
def apply_z_to_statevector(sv: np.ndarray, n_qubits: int, q: int) -> np.ndarray:
    psi = sv.reshape([2] * n_qubits)
    psi = np.swapaxes(psi, q, 0)
    psi[1, ...] *= -1 # multiply amplitudes of 1 states by -1
    psi = np.swapaxes(psi, 0, q)
    return psi.reshape(-1)

# apply random pauli gates to the circuit
def apply_local_pauli_noise(sv: np.ndarray, n_qubits: int, p: float, rng=None) -> np.ndarray:
    # if no seed is given use random seed
    if rng is None:
        rng = np.random.default_rng()

    # takes copy of statevector
    out = sv.copy()

    # randomly applies X, Y, Z paulis to qubit
    # higher p means higher error
    for q in range(n_qubits):
        r = rng.random() # random float from 0 to 1

        if r < (1 - p): # 1-p chance of applying identity
            continue

        elif r < (1 - p) + p/3: # p/3 chance of applying X
            out = apply_x_to_statevector(out, n_qubits, q)

        elif r < (1 - p) + 2*p/3: # p/3 chance of applying Y
            # applies Y pauli implicitly since Y=iXZ
            out = apply_z_to_statevector(out, n_qubits, q)
            out = apply_x_to_statevector(out, n_qubits, q)
            out = 1j * out

        else: # p/3 chance of applying Z
            out = apply_z_to_statevector(out, n_qubits, q)

    return out

# noise function
def haar_noise(sv_ideal, H_pad, W_pad, n_qubits, p=0.01, trials=100, seed=0):
    rng = np.random.default_rng(seed)
    Ys = []

    # runs multiple trials of noise to avg out at the end
    for _ in range(trials):
        sv_noisy = apply_local_pauli_noise(sv_ideal, n_qubits, p, rng=rng)
        Ys.append(sv_noisy.reshape(H_pad, W_pad))

    # take mean and standard deviation of the noisy Haar transformed images
    Ys = np.stack(Ys, axis=0)
    mean_Y = Ys.mean(axis=0)
    std_Y = np.abs(Ys).std(axis=0)

    return mean_Y, std_Y