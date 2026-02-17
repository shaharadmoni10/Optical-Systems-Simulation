# Optical_Communication_Task2
import math
import numpy as np  
from scipy.special import erfc, erfcinv
import matplotlib.pyplot as plt 



## שאלה 1
# system parameters
delta_f = 10e9 #רוחב פס
T=300
R_load = 50 
Responsivity = 1
Nf_dB = 5
BER_target = 1e-9
kB=1.38e-23

#import dat to linear and calculate Q factor
Nf_linear = 10**(Nf_dB/10)
Q = 6 #קירוב ידוע לדרישה

#noise calculation
sigma_thermal =math.sqrt((4*kB*T*delta_f*Nf_linear )/R_load)

#כעת אמרו לי שהרעש התרמי הוא המגביל אז אני מניח שלא משנה באיזה מצב שידור אנחנו, רק הרעש התרמי רלוונטי
#Q =  (I_1 - I_0) / (2*sigma_thermal) 
I1 = Q * 2 * sigma_thermal
P_in = I1 / (2*Responsivity) #linear
P_in_dB = 10*math.log10(P_in*1000)
print("the answer to question 1 is:",P_in_dB)


## שאלה 2
Q_vec = np.linspace(2,8,100)
BER = 1/2*erfc(Q_vec/np.sqrt(2))
I1 = Q_vec * 2 * sigma_thermal
P_avg_vec = I1 / (2*Responsivity)
P_avg_dB_vec = 10*np.log10(P_avg_vec*1000)
#draw the graph
plt.semilogy(P_avg_dB_vec, BER)
plt.xlabel('P_avg_dB')
plt.ylabel('BER')
plt.show()

## שאלה 3
BER_three=10**(-3)
Q_3 = erfcinv(2*BER_three)*math.sqrt(2)
N_bits = 100000
random_bits = np.random.randint(0, 2, N_bits)
I1 = Q_3 * 2 * sigma_thermal
random_current = I1 * random_bits
random_noise = np.random.normal(0, sigma_thermal, N_bits)
random_signal = random_current + random_noise
I_threshold = I1 / 2
decision_bool = random_signal > I_threshold
decision_bits = decision_bool.astype(int)
BER = np.mean(random_bits != decision_bits)
print("the answer to question 3 is:", BER)

#show histogram of random_signal
randon_0 = random_signal[random_bits == 0]
randon_1 = random_signal[random_bits == 1]
accuracy_percent = abs((BER - BER_three) / BER_three) * 100
plt.hist(randon_0, bins=100, alpha=0.6, label='0')
plt.hist(randon_1, bins=100, alpha=0.6, label='1')
plt.axvline(I_threshold, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
plt.legend(loc='upper right')
text_str = (f"Target BER: {BER_three}\n"        # היעד (למשל 1.0e-3)
            f"Simulated BER: {BER}\n"            # התוצאה בפועל
            f"the deviation is: {accuracy_percent:.2f}%")           # הסטייה באחוזים
plt.text(0.05, 0.95, text_str, 
         transform=plt.gca().transAxes,       # מיקום יחסי למסגרת הגרף
         fontsize=10,                         # גודל פונט
         verticalalignment='top',             # "תולה" את הטקסט מלמעלה
         bbox=dict(boxstyle='round',          # מסגרת עגולה ויפה
                   facecolor='white',         # רקע לבן
                   alpha=0.9,                 # שקיפות קלה (כדי לא להסתיר לגמרי)
                   edgecolor='gray'))         # צבע המסגרת
plt.show()  

## שאלה 4
M = 10
kA = 0.7
FA = kA * M + (1-kA)*(2-1/M)
I1_gain = M * I1
sigma_short_noise_square = 2 * 1.6e-19 * I1 * delta_f * FA * M**2
sigma_1 = math.sqrt((sigma_thermal**2 + sigma_short_noise_square))
I_threshold_new = (sigma_thermal * I1_gain) / (sigma_1 + sigma_thermal)
Q_new = I1_gain / (sigma_thermal + sigma_1)
BER_new = 1/2*erfc(Q_new/np.sqrt(2))

#noise calculation
noise_0_APD = np.random.normal(0, sigma_thermal, N_bits)
noise_1_APD = np.random.normal(0, sigma_1, N_bits)

#signal calculation
rx_signal_APD = np.zeros(N_bits)
rx_signal_APD[random_bits == 0] = 0 + noise_0_APD[random_bits == 0]
rx_signal_APD[random_bits == 1] = I1_gain + noise_1_APD[random_bits == 1] 
decisions_APD = (rx_signal_APD > I_threshold_new).astype(int)
BER_sim_APD = np.mean(random_bits != decisions_APD)
print("the answer to question 4 is:",BER_sim_APD)

plt.figure(figsize=(10, 6))
plt.hist(rx_signal_APD[random_bits==0], bins=100, alpha=0.6, color='blue', label='0')
plt.hist(rx_signal_APD[random_bits==1], bins=100, alpha=0.6, color='orange', label='1')
plt.axvline(I_threshold_new, color='red', linestyle='--', label='Threshold')
plt.yscale('log')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title(f'APD Histogram (M={M})'); plt.xlabel('Current [A]'); plt.ylabel('Count')
plt.show()

## שאלה 5
print("\n--- Question 5 Analysis ---")
# ספירת מספר השגיאות המדויק
num_errors = np.sum(random_bits != decisions_APD)
print(f"Total bits simulated: {N_bits}")
print(f"Total errors counted: {num_errors}")
if num_errors == 0:
    print("Conclusion: The BER is too low to be measured with this sample size.")
    print("Reliability: Low. We need significantly more bits to catch a rare error event.")
else:
    print("the answer to question 5 is:",num_errors/N_bits)