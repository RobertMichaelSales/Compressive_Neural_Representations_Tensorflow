import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,constrained_layout=True)

ax.tricontourf(i_volume.data[0:10052,0],i_volume.data[0:10052,1],i_values.data[0:10052,0],levels=25,vmin=-1,vmax=1)

plt.show()

fig,ax = plt.subplots(1,1,constrained_layout=True)

ax.tricontourf(o_volume.data[0:10052,0],o_volume.data[0:10052,1],o_values.data[0:10052,0],levels=25,vmin=-1,vmax=1)

plt.show()