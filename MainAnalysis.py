##%% -##################################
#             Imports
# -##################################
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciio
import f_Functions as f
import f_Graphics as g
from scipy import signal
import scipy.fft as fft

#%% -##################################
#       Lectura de datos
# -##################################
#TODO: Realice la lectura de datos, utilice los nombres proporcionados y la ruta del archivo debe ser generalizada.
str_FileName = os.path.join('EEGEpilepsyData.mat')
st_File = sciio.loadmat(str_FileName)
m_Data = np.double(st_File['v_MicroData'])
s_FsHz = np.double(st_File['s_SRate'])

#TODO: Verifique cómo se almacenan sus variables y, si es necesario, modifíquelas.
m_Data = m_Data.astype(np.int64)
m_Data = [dato for sublista in m_Data for dato in sublista]
#% Conjunto de datos utilizados para el análisis.
m_Data = m_Data[0:int(1200*s_FsHz)]
#m_Data = m_Data[int(1800*s_FsHz):int(3600*s_FsHz)]
v_Time = np.arange(0.0, np.size(m_Data)) / s_FsHz
print('data loaded')

##%% -##################################
#                Filtrado
# -##################################
#TODO: Aplique el filtro de respuesta infinita y grafique su señal resultante junto a la original.
#v_Time = np.arange(0.0, np.size(m_Data)) / s_FsHz
filt_FiltSOS = f.f_GetIIRFilter(s_FsHz, [80,500], [79, 501])
# filt_FiltSOS = f_GetIIRFilter(s_FsHz, [80, 2O0], [200, 500])
v_SigFilt = signal.sosfiltfilt(filt_FiltSOS, m_Data)
##
plt.style.use('seaborn-pastel')
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('EEG SIGNAL FILTERED', color = 'darkslategray', weight='bold')
ax[0].plot(v_Time, m_Data, color = 'teal',linewidth=1)
ax[0].set_title('Original Signal', color = 'darkslategray')
ax[1].plot(v_Time, v_SigFilt, color = 'goldenrod', linewidth=1)
ax[1].set_title('Filtered Signal', color = 'darkslategray')

ax[0].grid(linestyle = '--', linewidth = 0.5)
ax[1].grid(linestyle = '--', linewidth = 0.5)

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')

ax[0].set_xlabel('')
plt.show()
##%% -##################################
#                RMS
# -##################################
#TODO: Aplique un filtro RMS a su señal, esta no debe modificar la frecuencia de muestreo de la misma.
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = np.ones(window_size)/float(window_size)
  return np.sqrt(np.convolve(a2, window, 'valid'))

RMS = []
for i in range(0, len(v_SigFilt)):
    win = v_SigFilt[i:i+19]
    r = window_rms(win,20)
    RMS.append(r[0])
print('RMS calculated')

##%%Subplot
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('EEG SIGNAL FILTERED AND RMS', color = 'darkslategray', weight='bold')
ax[0].plot(v_Time, v_SigFilt, color = 'darkseagreen',linewidth=1)
#ax[1].plot(v_Time[0:len(v_SigRMS)], v_SigRMS, color = 'k', linewidth=1)
ax[1].plot(v_Time[0:len(RMS)], RMS, color = 'mediumseagreen', linewidth=1)
ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')
ax[0].set_title('Filtered Signal')
ax[1].set_title('RMS Signal')

ax[0].set_xlabel('')
ax[1].set_xlabel('Tiempo (s)')
plt.show()
##%% -##################################
# Detección y selección de picos
# -##################################
#TODO: Realice una detección de picos a su señal resultante de RMS y luego seleccione los picos correspondientes a eventos HFO
thres = (np.std(v_SigFilt))*5# Umbral de selección de picos
chanel_peaks, _ = signal.find_peaks(RMS, height = thres, distance = 6*s_FsHz)# Función que detecta picos por encima de umbral
RMS = np.array(RMS)

fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('EEG SIGNAL FILTERED AND CHANNEL PEAKS')
ax[0].plot(v_Time, v_SigFilt, color = 'slateblue',linewidth=1)
ax[1].plot(v_Time[0:len(RMS)], RMS, color = 'darkslateblue', linewidth=1)
ax[1].plot(v_Time[0:len(RMS)][chanel_peaks], RMS[chanel_peaks], ".r", color = 'black')
ax[1].hlines(thres,v_Time[0],v_Time[-1], color = 'silver', linewidth=1)

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')
ax[0].set_title('Filtered Signal')
ax[1].set_title('RMS Signal with Channel Peaks')

ax[0].set_xlabel('')
ax[1].set_xlabel('Tiempo (s)')
##%% Selección de picos.
selected_peaks = [] ## Picos seleccionados
selected_window = []
for i in range(len(chanel_peaks)):
    tres_ms = int(0.0031*s_FsHz)
    wind = RMS[chanel_peaks[i]-tres_ms:chanel_peaks[i]+tres_ms]
    is_there = np.where(wind < thres)
    if len(is_there[0]) == 0:
        selected_peaks.append(chanel_peaks[i])
        selected_window.append(wind)

print('De los ', len(chanel_peaks),' picos, el número de picos seleccionados es: ',len(selected_peaks))


fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('EEG SIGNAL FILTERED AND RMS SELECTED PEAKS', color ='darkslategray')

ax[0].plot(v_Time, v_SigFilt, color = 'slateblue',linewidth=1)
ax[1].plot(v_Time[0:len(RMS)], RMS, color = 'darkslateblue', linewidth=1)
ax[1].plot(v_Time[0:len(RMS)][selected_peaks], RMS[selected_peaks], ".r", color = 'gold')
ax[1].axhline(y=thres, color='silver', linestyle='-')

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')
ax[0].set_title('Filtered Signal')
ax[1].set_title('RMS Signal with Selected Peaks')

ax[0].set_xlabel('')
ax[1].set_xlabel('Tiempo (s)')

##% -##################################
#        Timepo frecuencia
# -##################################
#TODO: Realice un analisis de tiempo frecuencia solo a una ventana de Xs donde se haya detectado un HFO con una resolucion de 1Hz y 3 ciclos.
window_peak = v_SigFilt[selected_peaks[9]-(tres_ms*30):selected_peaks[9]+(tres_ms*30)]
window_time = v_Time[selected_peaks[9]-(tres_ms*30):selected_peaks[9]+(tres_ms*30)]
#buscar otro ripple cuando despues!!!!!!!!!
m_ConvMat, v_TimeArray, v_FreqTestHz = f.f_GaborTFTransform(window_peak, s_FsHz, 80, 500, 1, 3)

fig, ax = plt.subplots(2, 1,figsize=(8,8))
fig.suptitle('EEG SIGNAL FILTERED AND TIME-FREQUENCY')

ax[0].plot(window_time, window_peak, color = 'indigo',linewidth=1)

ConvMatPlot = m_ConvMat
m_ConvMatPlot = np.abs(m_ConvMat)
immat = ax[1].imshow(m_ConvMatPlot, cmap="twilight_shifted", interpolation='none',
                            origin='lower', aspect='auto',
                            extent=[v_TimeArray[0], v_TimeArray[-1],
                                    v_FreqTestHz[0], v_FreqTestHz[-1]],
                            vmin=-0, vmax=10000)

immat.set_clim(np.min(m_ConvMatPlot), np.max(m_ConvMatPlot) * 1.5)

ax[1].set_xlim([v_TimeArray[0], v_TimeArray[-1]])
fig.colorbar(immat, ax=ax[1])

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')
ax[0].set_title('Filtered Signal', color = 'darkslategray')
ax[1].set_title('Time-Frequency', color = 'darkslategray')

ax[0].set_xlabel('')
ax[1].set_xlabel('Tiempo (s)')
##%% -##################################
#           Visualización - fft
# -##################################
#ind = int(input('Ingrese el númeo del pico: '))
ind = 24
window_peak_fft = v_SigFilt[selected_peaks[ind]-(tres_ms*25):selected_peaks[ind]+(tres_ms*25)]
window_time_fft = v_Time[selected_peaks[ind]-(tres_ms*25):selected_peaks[ind]+(tres_ms*25)]
v_freq, v_psd = signal.welch(window_peak_fft, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft))

pos = np.where(v_freq > 500)[0][0]

fourier_y = v_freq[0:pos]
fourier_x = v_psd[0:pos]

plt.style.use('seaborn-pastel')
fig, ax = plt.subplots(2, 1, figsize=(8,8))
fig.suptitle('EEG SIGNAL FILTERED IN THE PEAK '+ str(ind)+' SELECTED.')

ax[0].plot(window_time_fft, window_peak_fft, color = 'palevioletred',linewidth=1)

ax[1].plot(fourier_y,fourier_x, color = 'maroon', linewidth=1)
ax[1].fill_between(fourier_y, fourier_x, color = 'maroon')

#ax[1].plot(v_Time[0:len(RMS)][selected_peaks], RMS[selected_peaks], ".r")
#ax[1].axhline(y=thres, color='r', linestyle='-')

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')
ax[0].set_title('Filtered Signal')
ax[1].set_title('Power Analysis')

ax[0].set_xlabel('')
ax[1].set_xlabel('Frequencia(Hz)')

##peak 16 as fast ripple.
window_peak_fft16 = v_SigFilt[selected_peaks[16]-(tres_ms*25):selected_peaks[16]+(tres_ms*25)]
window_time_fft16 = v_Time[selected_peaks[16]-(tres_ms*25):selected_peaks[16]+(tres_ms*25)]
v_freq16, v_psd16 = signal.welch(window_peak_fft16, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft16))

pos16 = np.where(v_freq16 > 500)[0][0]

fourier_y16 = v_freq16[0:pos16]
fourier_x16 = v_psd16[0:pos16]

#peak 5 as ripple.
window_peak_fft5 = v_SigFilt[selected_peaks[5]-(tres_ms*25):selected_peaks[5]+(tres_ms*25)]
window_time_fft5 = v_Time[selected_peaks[5]-(tres_ms*25):selected_peaks[5]+(tres_ms*25)]
v_freq5, v_psd5 = signal.welch(window_peak_fft5, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft5))

pos5 = np.where(v_freq5 > 500)[0][0]

fourier_y5 = v_freq5[0:pos5]
fourier_x5 = v_psd5[0:pos5]

##
#select peak 24 and 25 as ripple and fast ripple.

window_peak_fft24 = v_SigFilt[selected_peaks[24]-(tres_ms*25):selected_peaks[24]+(tres_ms*25)]
window_time_fft24 = v_Time[selected_peaks[24]-(tres_ms*25):selected_peaks[24]+(tres_ms*25)]
v_freq24, v_psd24 = signal.welch(window_peak_fft24, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft24))

pos24 = np.where(v_freq24 > 500)[0][0]

fourier_y24 = v_freq24[0:pos]
fourier_x24 = v_psd24[0:pos]

window_peak_fft25 = v_SigFilt[selected_peaks[25]-(tres_ms*25):selected_peaks[25]+(tres_ms*25)]
window_time_fft25 = v_Time[selected_peaks[25]-(tres_ms*25):selected_peaks[25]+(tres_ms*25)]
v_freq25, v_psd25 = signal.welch(window_peak_fft25, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft25))

pos25 = np.where(v_freq25 > 500)[0][0]

fourier_y25 = v_freq25[0:pos25]
fourier_x25 = v_psd25[0:pos25]

window_time_fft_double = np.concatenate((window_time_fft25, window_time_fft24))
window_peak_fft_double = np.concatenate((window_peak_fft25, window_peak_fft24))
window_time_fft_double = np.linspace(window_time_fft_double[0],window_time_fft_double[-1],len(window_peak_fft_double))
v_freq_double, v_psd_double = signal.welch(window_peak_fft_double, s_FsHz, 'flattop',nperseg = None, nfft=len(window_peak_fft_double))

posd = np.where(v_freq_double > 500)[0][0]

fourier_y_double = v_freq_double[0:posd]
fourier_x_double = v_psd_double[0:posd]

window_time_fft_d = np.concatenate((window_time_fft_double[0:400],window_time_fft_double[2400:2800]))
window_peak_fft_d = np.concatenate((window_peak_fft_double[0:400],window_peak_fft_double[2400:2800]))

plt.style.use('seaborn-pastel')
fig, ax = plt.subplots(2, 1)
fig.suptitle('EEG SIGNAL FILTERED IN THE PEAK '+ str(16)+' SELECTED. \n FAST RIPPLE', color = 'darkslategrey')

ax[0].plot(window_time_fft_double, window_peak_fft_double,linewidth=1, color = 'lightseagreen')

ax[1].plot(fourier_y_double,fourier_x_double, linewidth=1, color = 'steelblue')
ax[1].fill_between(fourier_y_double, fourier_x_double, color = 'steelblue')

ax[0].set_ylabel('Amplitud')
ax[1].set_ylabel('Amplitud')

ax[0].set_xlabel('')
ax[1].set_xlabel('Frequencia(Hz)')
##
plt.style.use('seaborn-pastel')
#plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
fig, ax = plt.subplots(3, 2,figsize=(8,8))
#fig.suptitle('EEG SIGNAL FILTERED WITH DIFFERENT TYPES OF PEAKS', color = 'darkslategray')

ax[0,0].plot(window_time_fft16, window_peak_fft16,linewidth=1, color = 'lightseagreen')
ax[0,1].plot(fourier_y16,fourier_x16, linewidth=1, color = 'steelblue')
ax[0,1].fill_between(fourier_y16, fourier_x16, color = 'steelblue')
ax[0,1].plot(326.795,3.23232, "o", color = 'darkslateblue')
ax[0,0].set_ylabel('Amplitud')
ax[0,1].set_ylabel('Amplitud')
ax[0,0].set_xlabel('')
ax[0,1].set_xlabel('Frequencia(Hz)')
ax[0,0].set_title('Filtered Signal Fast Ripple',color = 'darkslategray')
ax[0,1].set_title('Power Spectral Analysis',color = 'darkslategray')

ax[1,0].plot(window_time_fft5, window_peak_fft5,linewidth=1, color = 'palevioletred')
ax[1,1].plot(fourier_y5,fourier_x5, linewidth=1, color = 'lightpink')
ax[1,1].fill_between(fourier_y5, fourier_x5, color = 'lightpink')
ax[1,1].plot(301.941,2.2003, "o", color = 'mediumvioletred')
ax[1,0].set_ylabel('Amplitud')
ax[1,1].set_ylabel('Amplitud')
ax[1,0].set_xlabel('')
ax[1,1].set_xlabel('Frequencia(Hz)')
ax[1,0].set_title('Filtered Signal Ripple',color = 'darkslategray')
ax[1,1].set_title('Power Spectral Analysis',color = 'darkslategray')


ax[2,0].plot(window_time_fft_double, window_peak_fft_double,linewidth=1, color = 'darkmagenta')
ax[2,1].plot(fourier_y_double,fourier_x_double, linewidth=1, color = 'plum')
ax[2,1].fill_between(fourier_y_double, fourier_x_double, color = 'plum')
ax[2,1].plot(216.123,3.5035, "o", color = 'mediumorchid')
ax[2,1].plot(429.018,2.56757, "o", color = 'mediumorchid')
ax[2,0].set_ylabel('Amplitud')
ax[2,1].set_ylabel('Amplitud')
ax[2,0].set_xlabel('')
ax[2,1].set_xlabel('Frequencia(Hz)')
ax[2,0].set_title('Filtered Signal Both Fast and Slow Ripple', color = 'darkslategray')
ax[2,1].set_title('Power Spectral Analysis', color = 'darkslategray')
fig.set_tight_layout(True)