import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

# N = 51
# w = signal.hann(N)
#
# A = fft(w, 2048) / (len(w)/2.0)
# response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

def peak_detection(pulse):
    freq = np.linspace(-1, 1, np.shape(pulse)[0])
    response = np.array([i for i in pulse[:,0]])

    num=9#移動平均の個数
    b=np.ones(num)/num

    response=np.convolve(response, b, mode='same')
    # response = [i for i in original_pulse[:,0]]
    # 極大値のインデックスを取得
    maxId = signal.argrelmax(response)
    # maxId = signal.argrelextrema(response, np.greater)

    # 極小値のインデックスを取得
    minId = signal.argrelmin(response)
    # minId = signal.argrelextrema(response, np.less)



    print("numeric result", len(maxId[0]), len(minId[0]))

    plt.plot(freq, response)
    plt.plot(freq[maxId], response[maxId], "ro")
    plt.plot(freq[minId], response[minId], "bo")
    plt.axis("tight")
    plt.show()

if __name__ == "__main__":
    original_pulse = np.loadtxt("30bpmresult.csv", delimiter=",")
    peak_detection(original_pulse)
    original_pulse = np.loadtxt("60bpmresult.csv", delimiter=",")
    peak_detection(original_pulse)
    original_pulse = np.loadtxt("90bpmresult.csv", delimiter=",")
    peak_detection(original_pulse)
    original_pulse = np.loadtxt("120bpmresult.csv", delimiter=",")
    peak_detection(original_pulse)
