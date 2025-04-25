"""
==========
WIR_Gumbel_1
==========
Script zum Erzeugen eines Wahrscheinlichkeitspapiers für die Gumbelverteilung

Copyright (c) 2022 Dr.-Ing. Wilhelm Riebe (wfhsoft@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory
#
#Liste der Wahrscheinlichkeiten
#1.Spalte: Quantile der Gumbelverteilung zur Anzeige an der Y-Achse
#2.Spalte: Beschriftung der Y-Achsen
pobabilities = OrderedDict(
    [
        (0.001, 'P'),
        (0.005,  'P'),
        (0.01, 'P'), #11
        (0.02, ''), 
        (0.03, ''), 
        (0.04, ''), 
        (0.05,  'P'),
        (0.06,  ''),
        (0.07,  ''),
        (0.08,  ''),
        (0.09,  ''),
        (0.1, 'P'), #10
        (0.11,  ''),
        (0.12,  ''),
        (0.13,  ''),
        (0.14,  ''),
        (0.15,  '*'),
        (0.16,  ''),
        (0.17,  ''),
        (0.18,  ''),
        (0.19,  ''),
        (0.2, 'P'), #20
        (0.21,  ''),
        (0.22,  ''),
        (0.23,  ''),
        (0.24,  ''),
        (0.25,  '*'),
        (0.26,  ''),
        (0.27,  ''),
        (0.28,  ''),
        (0.29,  ''),
        (0.3, 'P'), #30
        (0.31,  ''),
        (0.32,  ''),
        (0.33,  ''),
        (0.34,  ''),
        (0.35,  '*'),
        (0.36,  ''),
        (0.36788,  'mod'),
        (0.37,  ''),
        (0.38,  ''),
        (0.39,  ''),
        (0.4, 'P'), #40
        (0.41,  ''),
        (0.42,  ''),
        (0.43,  ''),
        (0.44,  ''),
        (0.45,  '*'),
        (0.46,  ''),
        (0.47,  ''),
        (0.48,  ''),
        (0.49,  ''),
        (0.5, 'PT'), #50
        (0.51, ''),
        (0.52, ''),
        (0.53, ''),
        (0.54, ''),
        (0.55, '*'),
        (0.56, ''),
        (0.57, ''),
        (0.5772, 'Mittel'),
        (0.58, ''),
        (0.59, ''),
        (0.6, 'P'), #60
        (0.61,  ''),
        (0.62,  ''),
        (0.63,  ''),
        (0.64,  ''),
        (0.65,  '*'),
        (0.66,  ''),
        (0.67,  ''),
        (0.68,  ''),
        (0.69,  ''),
        (0.7, 'P'), #70
        (0.71,  ''),
        (0.72,  ''),
        (0.73,  ''),
        (0.74,  ''),
        (0.75,  '*'),
        (0.76,  ''),
        (0.77,  ''),
        (0.78,  ''),
        (0.79,  ''),
        (0.8, 'PT'), #80
        (0.81,  ''),
        (0.82,  ''),
        (0.83,  ''),
        (0.84,  ''),
        (0.85,  '*'),
        (0.86,  'T'),
        (0.87,  ''),
        (0.88,  ''),
        (0.89,  ''),
        (0.9, 'PT'),#90
        (0.91,  ''),
        (0.92,  ''),
        (0.93,  ''),
        (0.933,  'T'),
        (0.94,  ''),
        (0.95,  'PT'),
        (0.955,   ''),
        (0.96,  'PT'),
        (0.965,  ''),
        (0.9667,  'T'),
        (0.97,   'P'),
        (0.972,   ''),
        (0.974,   ''),
        (0.976,   ''),
        (0.978,   ''),
        (0.98,  'PT'), #98
        (0.982,  ''),
        (0.984,  ''),
        (0.986,  ''),
        (0.988,  ''),
        (0.99, 'PT'), #99
        (0.991,  ''),
        (0.992,  ''),
        (0.993,  ''),
        (0.99334,  'T'),
        (0.994,  ''),
        (0.995,  'PT'),
        (0.996,  'PT'),
        (0.99667,  'T'),
        (0.997,  'P'),
        (0.9975,  'T'),
        (0.998,  'PT'),
        (0.9981,  ''),
        (0.9982,  ''),
        (0.9983,  ''),
        (0.9984,  ''),
        (0.9985,  ''),
        (0.998571,  'T'),
        (0.9986,  ''),
        (0.9987,  ''),
        (0.9988,  ''),
        (0.9989,  ''),
        (0.999, 'PT'),
        (0.9991, ''),
        (0.9992, ''),
        (0.9993, 'P'),
        (0.9994, ''),
        (0.9995,  ''),
       
    ])
#
#Wahrscheinlichkeitsfunktion
def fktProb(prob):
        return np.log(-np.log(prob))
#
#
#
#Settings
#-------------------------------------------------------------------------
delta =0.0
linewidth_major=0.9
linewidth_major_T=1.0
linewidth_minor=0.3
linewidth_mu=1.0
fontsize_major= 10
fontsize_minor=10
fontsize_mu=9
linelength=230 #230
probmin=0.0001
probmax=0.9995
family_fonts='monospace'
fig=plt.figure(figsize=(16, 11)) #inches (16, 11)
ax = plt.subplot(1, 1, 1)
#-------------------------------------------------------------------------
#
ax.set_ylim(delta +fktProb(probmin), delta + fktProb(probmax))
X, Y = np.linspace(0, linelength, 10), np.zeros(10)

plt.title('Wahrscheinlichkeitsnetz (Gumbelverteilung)', fontsize=16)
plt.ylabel('Plotting Positions nach Gringorten', fontsize=16)
#
#Grid and ticks
plt.yticks([])
major_ticks = np.arange(0,linelength+1, 10)
minor_ticks = np.arange(0, linelength+1, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels([])
plt.grid(axis='x',  which='both')
for xmaj in ax.xaxis.get_majorticklocs():
  ax.axvline(x=xmaj, ls='-', lw=linewidth_major,  color='black')
for xmin in ax.xaxis.get_minorticklocs():
 ax.axvline(x=xmin, ls='-', lw=linewidth_minor,   color='black')
#
#Erstellen des Koordinatensystems
#-------------------------------------------------------------------------
box = ax.get_position()
ax.set_position([box.x0-0.05, box.y0, box.width, box.height]) #Verschiebung nach rechts
xmin, xmax, ymin, ymax = plt.axis()
plt.text(xmin-11,ymin+0.7,'Entwurf: Dr.-Ing. Wilhelm Riebe \nPython3.10',   fontsize=8)
x1pos0=xmin+ 42.0 #linke Y-Achse
y2pos1=xmin +980 #978
y2pos2=y2pos1-10
y1pos0=-2.0
y1pos1=y1pos0-0.2
#-------------------------------------------------------------------------
reference_transform = blended_transform_factory(ax.transAxes, ax.transData)
for i, (prob, labelstyle) in enumerate(pobabilities.items()):
    p=prob*100
    T = 1.0/(1.0 -float(prob))
    if  labelstyle =='Mittel':
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=1.5, color='darkblue')
        ax.annotate('$\mu$', xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(y2pos1, y1pos0), textcoords='offset points', color="darkblue",
        fontsize=fontsize_major, ha="right", family=family_fonts)
    elif  labelstyle =='mod':
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=1.5, color='darkblue')
        ax.annotate('mod', xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(y2pos1+10, y1pos0), textcoords='offset points', color="darkblue",
        fontsize=fontsize_major, ha="right", family=family_fonts)
    elif labelstyle =='*':
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=linewidth_major, color='black')
        ax.annotate('', xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(x1pos0, y1pos0), textcoords='offset points', color="black",
        fontsize=fontsize_minor, ha="right", family=family_fonts) 
    elif labelstyle =='':
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=linewidth_minor, color='black')
        ax.annotate('', xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(x1pos0, y1pos0), textcoords='offset points', color="black",
        fontsize=fontsize_minor, ha="right", family=family_fonts) 
    elif  labelstyle =='P':  
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=linewidth_major, color='black')
        ax.annotate('{:02.2f}'.format(p), xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(x1pos0, y1pos0), textcoords='offset points', color="black",
        fontsize=fontsize_minor, ha="right", family=family_fonts)  
    elif  labelstyle =='PT':  
        ax.plot(X, Y+fktProb(prob)+delta, linestyle=  (0, (5, 1)), linewidth=linewidth_major_T, color='darkblue')
        ax.annotate('{:02.2f}'.format(p), xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(x1pos0, y1pos0), textcoords='offset points', color="black",
        fontsize=fontsize_minor, ha="right", family=family_fonts)  
        ax.annotate('{:.0f}'.format(T), xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(y2pos2, y1pos1), textcoords='offset points', color="darkblue",
        fontsize=fontsize_minor, ha="left", family=family_fonts)   
    elif  labelstyle =='T':  
        ax.plot(X, Y+fktProb(prob)+delta, linestyle=  (0, (5, 1)), linewidth=linewidth_major_T, color='darkblue')
        ax.annotate('{:.0f}'.format(T), xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(y2pos2, y1pos1), textcoords='offset points', color="darkblue",
        fontsize=fontsize_minor, ha="left", family=family_fonts)     
    else: 
        ax.plot(X, Y+fktProb(prob)+delta, linestyle= (0, ()), linewidth=linewidth_minor, color='red') #dann muss was falsch sein
        ax.annotate('', xy=(0.0, fktProb(prob)+delta ), xycoords=reference_transform,
        xytext=(x1pos0, y1pos0), textcoords='offset points', color="black",
        fontsize=fontsize_minor, ha="right", family=family_fonts)   
#
plt.tight_layout()
plt.savefig("WIR_Gumbelpapier_2.pdf")
plt.show()
