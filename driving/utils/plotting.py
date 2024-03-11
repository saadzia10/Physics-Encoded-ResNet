import pickle,os
import matplotlib.pyplot as plt

# data_filename = "torcs_a2c_accel_multvar_2018-06-01_004341.335@ep_400_scored_328.pickle"
data_filename = "torcs_a2c_accel_multvar_2018-06-01_163715.008%40ep_1499_scored_1613.pickle"
with open( os.getcwd() + "/trained_models/" + data_filename, "rb") as pf:
    data = pickle.load( pf)

plt.plot( [ i for i in range( len( data))], data)

plt.show()
