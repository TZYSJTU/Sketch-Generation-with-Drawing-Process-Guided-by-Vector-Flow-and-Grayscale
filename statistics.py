import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    img_path   = './Patch/014.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    mean = np.zeros(img.shape[1])
    var  = np.zeros(img.shape[1])
    peak_mean = np.zeros(img.shape[1])
    peak_var  = np.zeros(img.shape[1])
    valley_mean = np.zeros(img.shape[1])
    valley_var = np.zeros(img.shape[1])
    period = np.zeros(img.shape[1])


    ###########-------Column-------###############
    for i in range (img.shape[1]):
        column = img[:,i]
        # plt.plot(column)
        # plt.show()

        mean[i] = np.mean(column)
        var[i]  = np.var(column)
        print("Column:",i,"Mean:{:.1f}".format(mean[i]),"Var:{:.1f}".format(var[i]))

    print("Mean:{:.1f}".format(mean.mean()),"Var:{:.1f}".format(var.mean()))


    ##########---------period--------###############
    for i in range (img.shape[1]):
        column = img[:,i]
        peak_list = []
        valley_list = []
        period_list = []

        peak_last = 255
        valley_last = 0
        for j in range(column.shape[0]):
            if j>0 and j<column.shape[0]-1:
                # peak
                if column[j]>column[j-1] and column[j]>column[j+1]:
                    if valley_last+ 25 < column[j]:#####################################
                        peak_list.append(column[j])
                        peak_last = column[j]
                # valley
                else:
                    if column[j]<column[j-1] and column[j]<column[j+1]:
                        if column[j]+ 25 < peak_last:###################################
                            valley_list.append(column[j])
                            valley_last = column[j]
                            period_list.append(j)
                

                                              
        peak_list = np.array(peak_list)
        valley_list = np.array(valley_list)
        period_list = np.array(period_list)

        peak_mean[i] = peak_list.mean()
        peak_var[i]  = peak_list.var()
        valley_mean[i] = valley_list.mean()
        valley_var[i]  = valley_list.var()
        period[i] = (period_list[-1]-period_list[0])/len(period_list-1)

        print("\nColumn:{}".format(i))
        print(
            "  Peak number:{:.1f}  ".format(len(peak_list)),
            "  Peak Mean:{:.1f}  ".format(peak_mean[i]),
            "  Peak Var:{:.1f}".format(peak_var[i]),
        )
        print(
            "  Valley number:{:.1f}".format(len(valley_list)),
            "  Valley Mean:{:.1f}".format(valley_mean[i]),
            "  Valley Var:{:.1f}".format(valley_var[i]),
        )
        print("Period:{:.1f}".format(period[i]))

    print(
        "\nPeak Mean:{:.1f}".format(peak_mean.mean()),
        "Peak Var:{:.1f}".format(peak_var.mean()),
        "Valley Mean:{:.1f}".format(valley_mean.mean()),
        "Valley Var:{:.1f}".format(valley_var.mean()),
        "Period Mean:{:.1f}".format(period.mean()),
        "Period Var:{:.1f}".format(period.var())
        )
    
    ########################################################
    ###########----------for all period-------############
    ########################################################
    periods = []
    peak_list = []
    valley_list = []
    radios = int(round(period.mean()/2))

    for i in range (img.shape[1]):
        column = img[:,i]
        peak_last = 255
        valley_last = 0

        for j in range(column.shape[0]):
            if j>0 and j<column.shape[0]-1:
                # peak
                if column[j]>column[j-1] and column[j]>column[j+1]:
                    if valley_last+ 25 < column[j]:#####################################
                        peak_list.append(column[j])
                        peak_last = column[j]
                # valley
                else:
                    if column[j]<column[j-1] and column[j]<column[j+1]:
                        if column[j]+ 25 < peak_last:###################################
                            valley_list.append(column[j])
                            valley_last = column[j]

                            if j-radios >= 0 and j+radios <= column.shape[0]-1:
                                t = column[j-radios:j+radios+1]
                                t = t[np.newaxis,:]
                                periods.append(t)
                               
    period = np.concatenate((periods),axis=0)
    period_stastic_mean = np.mean(period,axis=0)
    period_stastic_var  = np.var(period,axis=0)
    plt.plot(period_stastic_mean)
    plt.plot(period_stastic_var)
    plt.show()
    print("period_stastic_mean\n",period_stastic_mean)
    print("period_stastic_var\n",period_stastic_var)










            
                              






