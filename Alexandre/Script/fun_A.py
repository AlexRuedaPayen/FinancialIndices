import os,sys
sys.path.append(os.getcwd())
#sys.path.append("../..")

def fun_A():

    from Alexandre.Class.history import history

    RUI_PA=history(name='RUI.PA')
    VPK_AS=history(name='VPK.AS')
    BP_L=history(name="BP.L")
    SHELL_AS=history(name="SHELL.AS")
    TTE_PA=history(name="TTE.PA")
    XOM=history(name="XOM")

    RUI_PA.MA()
    VPK_AS.MA()
    BP_L.MA()
    SHELL_AS.MA()
    TTE_PA.MA()
    XOM.MA()

    RUI_PA.derivative_rate()
    VPK_AS.derivative_rate()
    BP_L.derivative_rate()
    SHELL_AS.derivative_rate()
    TTE_PA.derivative_rate()
    XOM.derivative_rate()

    print(RUI_PA.derivative_rate)

    RUI_PA.plot()

if __name__=='__main__':
    fun_A()