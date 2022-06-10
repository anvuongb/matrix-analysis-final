import numpy as np
import matplotlib
matplotlib.use('agg') # fix pyplot hangs in wsl2 https://github.com/matplotlib/matplotlib/issues/22385
import matplotlib.pyplot as plt
from IPython.display import clear_output

# HALS_Accel_L30
HALS_Accel_L30_e_time = np.load("./HALS_Accel_L30/weights/e_time.npy")
HALS_Accel_L30_norm_error = np.load("./HALS_Accel_L30/weights/norm_error.npy")

# HALS_Accel_L60
HALS_Accel_L60_e_time = np.load("./HALS_Accel_L60/weights/e_time.npy")
HALS_Accel_L60_norm_error = np.load("./HALS_Accel_L60/weights/norm_error.npy")

# HALS_L30
HALS_L30_e_time = np.load("./HALS_L30/weights/e_time.npy")
HALS_L30_norm_error = np.load("./HALS_L30/weights/norm_error.npy")

# HALS_L60
HALS_L60_e_time = np.load("./HALS_L60/weights/e_time.npy")
HALS_L60_norm_error = np.load("./HALS_L60/weights/norm_error.npy")

# MU_Accel_L30
MU_Accel_L30_e_time = np.load("./MU_Accel_L30/weights/e_time.npy")
MU_Accel_L30_norm_error = np.load("./MU_Accel_L30/weights/norm_error.npy")

# MU_Accel_L60
MU_Accel_L60_e_time = np.load("./MU_Accel_L60/weights/e_time.npy")
MU_Accel_L60_norm_error = np.load("./MU_Accel_L60/weights/norm_error.npy")

# MU_L30
MU_L30_e_time = np.load("./MU_L30/weights/e_time.npy")
MU_L30_norm_error = np.load("./MU_L30/weights/norm_error.npy")

# MU_L60
MU_L60_e_time = np.load("./MU_L60/weights/e_time.npy")
MU_L60_norm_error = np.load("./MU_L60/weights/norm_error.npy")

# e_min = np.min(np.array([HALS_Accel_L30_e_time, HALS_Accel_L60_e_time,
#                          HALS_L30_e_time, HALS_L60_e_time,
#                          MU_Accel_L30_e_time, MU_Accel_L60_e_time,
#                          MU_L30_e_time, MU_L60_e_time   
#                         ]))
e_min = np.min(HALS_Accel_L30_e_time.tolist()+ HALS_Accel_L60_e_time.tolist()+
                HALS_L30_e_time.tolist()+ HALS_L60_e_time.tolist()+
                MU_Accel_L30_e_time.tolist()+ MU_Accel_L60_e_time.tolist()+
                MU_L30_e_time.tolist()+ MU_L60_e_time.tolist()   
            )

HALS_Accel_L30_E = [(e-e_min)/(HALS_Accel_L30_norm_error[0]-e_min) for e in HALS_Accel_L30_norm_error]
HALS_Accel_L60_E = [(e-e_min)/(HALS_Accel_L60_norm_error[0]-e_min) for e in HALS_Accel_L60_norm_error]
HALS_L30_E = [(e-e_min)/(HALS_L30_norm_error[0]-e_min) for e in HALS_L30_norm_error]
HALS_L60_E = [(e-e_min)/(HALS_L60_norm_error[0]-e_min) for e in HALS_L60_norm_error]

MU_Accel_L30_E = [(e-e_min)/(MU_Accel_L30_norm_error[0]-e_min) for e in MU_Accel_L30_norm_error]
MU_Accel_L60_E = [(e-e_min)/(MU_Accel_L60_norm_error[0]-e_min) for e in MU_Accel_L60_norm_error]
MU_L30_E = [(e-e_min)/(MU_L30_norm_error[0]-e_min) for e in MU_L30_norm_error]
MU_L60_E = [(e-e_min)/(MU_L60_norm_error[0]-e_min) for e in MU_L60_norm_error]

fig = plt.figure()
plt.plot(HALS_Accel_L30_e_time, HALS_Accel_L30_E, label="HALS_Accel_L30_E")
plt.plot(HALS_Accel_L60_e_time, HALS_Accel_L60_E, label="HALS_Accel_L60_E")
plt.plot(HALS_L30_e_time, HALS_L30_E, label="HALS_L30_E")
plt.plot(HALS_L60_e_time, HALS_L60_E, label="HALS_L60_E")
plt.plot(MU_Accel_L30_e_time, MU_Accel_L30_E, label="MU_Accel_L30_E")
plt.plot(MU_Accel_L60_e_time, MU_Accel_L60_E, label="MU_Accel_L60_E")
plt.plot(MU_L30_e_time, MU_L30_E, label="MU_L30_E")
plt.plot(MU_L60_e_time, MU_L60_E, label="MU_L60_E")
plt.legend()
plt.xlabel("time (seconds)")
plt.ylabel("E(t)")
plt.tight_layout()
fig.savefig("Et.png")

fig = plt.figure()
plt.plot(HALS_Accel_L30_e_time, HALS_Accel_L30_E, label="HALS_Accel_L30_E")
plt.plot(HALS_Accel_L60_e_time, HALS_Accel_L60_E, label="HALS_Accel_L60_E")
plt.plot(HALS_L30_e_time, HALS_L30_E, label="HALS_L30_E")
plt.plot(HALS_L60_e_time, HALS_L60_E, label="HALS_L60_E")
plt.plot(MU_Accel_L30_e_time, MU_Accel_L30_E, label="MU_Accel_L30_E")
plt.plot(MU_Accel_L60_e_time, MU_Accel_L60_E, label="MU_Accel_L60_E")
plt.plot(MU_L30_e_time, MU_L30_E, label="MU_L30_E")
plt.plot(MU_L60_e_time, MU_L60_E, label="MU_L60_E")
# plt.legend()
plt.xlabel("time (seconds)")
plt.ylabel("E(t)")
plt.xlim(-1, 30)
plt.tight_layout()
fig.savefig("Et_limit.png")

fig = plt.figure()
plt.plot(range(len(HALS_Accel_L30_e_time)), HALS_Accel_L30_E, label="HALS_Accel_L30_E")
plt.plot(range(len(HALS_Accel_L60_e_time)), HALS_Accel_L60_E, label="HALS_Accel_L60_E")
plt.plot(range(len(HALS_L30_e_time)), HALS_L30_E, label="HALS_L30_E")
plt.plot(range(len(HALS_L60_e_time)), HALS_L60_E, label="HALS_L60_E")
plt.plot(range(len(MU_Accel_L30_e_time)), MU_Accel_L30_E, label="MU_Accel_L30_E")
plt.plot(range(len(MU_Accel_L60_e_time)), MU_Accel_L60_E, label="MU_Accel_L60_E")
plt.plot(range(len(MU_L30_e_time)), MU_L30_E, label="MU_L30_E")
plt.plot(range(len(MU_L60_e_time)), MU_L60_E, label="MU_L60_E")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("E(k)")
plt.tight_layout()
fig.savefig("Ek.png")

