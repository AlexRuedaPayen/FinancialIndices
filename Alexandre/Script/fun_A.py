import os,sys
print(os.getcwd())
sys.path.append("../..")

def fun_A():

    from Alexandre.Class.Stock import Stock

    RUI_PA=Stock(name='RUI.PA')
    VPK_AS=Stock(name='VPK.AS')
    BP_L=Stock(name="BP.L")
    SHELL_AS=Stock(name="SHELL.AS")
    TTE_PA=Stock(name="TTE.PA")
    XOM=Stock(name="XOM")

    print('ok')

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

fun_A()