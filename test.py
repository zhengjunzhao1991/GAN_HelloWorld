import matplotlib.pyplot as plt

import tflearn
import tensorflow as tf


# fig = plt.figure(figsize=(3, 3))
#
# fig.set_figheight(10)
# fig.set_figwidth(30)
# f, axs = plt.subplots(2,2,figsize=(15,15))
# ax = fig.add_subplot(1, 1, 1, frameon=False)
# ax.set_xlim(-0.015, 1.515)
# ax.set_ylim(-0.01, 1.01)
# ax.set_xticks([0, 0.3, 0.4, 1.0, 1.5])
# #增加0.35处的刻度并不标注文本，然后重新标注0.3和0.4处文本
# ax.set_xticklabels([0.0, "", "", 1.0, 1.5])
# ax.set_xticks([0.35], minor=True)
# ax.set_xticklabels(["0.3 0.4"], minor=True)
#
# #上述设置只是增加空间，并不想看到刻度的标注，因此次刻度线不予显示。
# for line in ax.xaxis.get_minorticklines():
#     line.set_visible(False)
# ax.grid(True)


# plt.show()
print(help(tflearn.DNN))