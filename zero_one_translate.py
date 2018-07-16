import numpy as np

def calculate_shuuki(original_pulse):
    out_arr = []
    for i in original_pulse[:,0]:
        if i < 0.5:
            out_arr.append(0)
        elif i >= 0.5:
            out_arr.append(1)

    prevval = out_arr[0]
    counter = 0
    counter_arr = []

    for i in out_arr[1:]:
        if prevval == i:
            counter += 1
        else:
            if counter > 5:
                counter_arr.append(counter)
            counter = 1
            prevval = i

    print(counter_arr)

    sum = np.sum(counter_arr) * 2 / float(len(counter_arr))
    print(sum)


if __name__ == "__main__":
    original_pulse = np.loadtxt("30bpmresult.csv", delimiter=",")
    calculate_shuuki(original_pulse)
    original_pulse = np.loadtxt("60bpmresult.csv", delimiter=",")
    calculate_shuuki(original_pulse)
    original_pulse = np.loadtxt("90bpmresult.csv", delimiter=",")
    calculate_shuuki(original_pulse)
    original_pulse = np.loadtxt("120bpmresult.csv", delimiter=",")
    calculate_shuuki(original_pulse)
