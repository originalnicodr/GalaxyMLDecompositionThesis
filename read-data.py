import joblib
import numpy as np

title = "Internal evaluation: attributes='eps', 'eps_r' n_clusters=3"

data = joblib.load(title+".data")

normalized_star_energy, eps, eps_r, label = np.array(data).T


#import sys
#from PyQt5 import QtWidgets
#QtWidgets.QApplication(sys.argv)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print(type(normalized_star_energy))
print(type(eps))
print(type(eps_r))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(normalized_star_energy, eps, eps_r, c=label)
ax.set_title(title)

fig.savefig(title+'.png')