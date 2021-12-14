from matplotlib import pyplot as plt

rews = [-145400.0, -993970.0, -16900.0, -1000000.0, -7820.0, -8805.0, -9825.0, -8830.0, -9825.0, -8830.0, -12850.0, -9825.0, -15875.0, -7805.0, -9865.0, -7805.0, -7805.0, -12850.0, -16900.0, -12850.0, -16900.0, -12850.0, -12850.0, -12850.0, -15875.0, -8805.0, -15875.0, -15875.0, -12850.0, -9825.0, -8805.0, -8805.0, -12850.0, -9825.0, -9825.0, -16900.0, -12850.0, -15875.0, -15875.0, -15875.0, -12850.0, -16900.0, -15875.0, -9825.0, -8805.0, -15875.0, -8845.0, -12850.0, -8845.0, -15875.0, -6810.0, -15875.0, -9830.0, -12850.0, -12850.0, -15875.0, -12850.0, -15875.0, -12850.0, -12850.0, -12850.0, -16900.0, -12850.0, -12850.0, -12850.0, -6810.0, -7805.0, -15875.0, -16900.0, -12850.0, -16900.0, -12850.0, -15875.0, -9830.0, -7830.0, -15875.0, -12850.0, -12850.0, -15875.0, -15875.0, -7830.0, -15875.0, -9825.0, -12850.0, -15875.0, -7805.0, -8830.0, -12850.0, -16900.0, -9825.0, -16900.0, -9825.0, -15875.0, -7830.0, -12850.0, -12850.0, -7830.0, -8845.0, -15875.0, -12850.0, -12850.0, -8805.0, -9825.0, -9825.0, -16900.0, -15875.0, -10820.0, -12850.0, -9840.0, -8805.0, -21690.0, -13845.0, -28670.0, -43730.0, -49590.0, -67380.0, -24665.0, -8800.0, -8655.0, -68375.0, -34680.0, -28675.0, -57445.0, -32570.0, -77360.0, -38575.0, -13845.0, -33600.0, -38575.0, -49705.0, -16900.0, -15875.0, -9825.0, -15875.0, -16900.0, -12850.0, -9830.0, -15875.0, -12850.0, -7805.0, -12850.0, -12850.0, -12850.0, -12850.0, -9865.0, -15875.0, -9825.0, -12850.0, -12850.0, -12850.0, -15875.0, -16900.0, -12850.0, -15875.0, -12850.0, -12850.0, -12850.0, -15875.0, -9830.0, -12850.0, -15875.0, -8830.0, -16900.0, -13845.0, -8830.0, -12850.0, -8830.0, -12850.0, -15875.0, -15875.0, -16870.0, -16900.0, -16900.0, -12850.0, -7830.0, -16900.0, -12850.0, -12850.0, -16900.0, -15875.0, -9840.0, -14840.0, -11815.0, -8800.0, -62430.0, -20880.0, -17895.0, -28810.0, -14840.0, -18860.0, -16870.0, -9840.0, -17825.0, -9820.0, -75490.0, -44560.0, -19855.0, -8800.0, -10790.0, -18890.0, -33785.0, -34475.0, -11785.0, -34740.0, -60430.0, -61295.0, -70340.0, -10790.0, -39540.0, -9820.0, -8800.0, -120120.0, -50280.0, -21490.0, -12825.0, -52320.0, -22610.0, -22600.0, -29600.0, -51130.0, -46265.0, -29585.0, -12825.0, -51350.0, -30610.0, -45445.0, -28325.0, -195640.0, -34315.0, -80645.0, -12800.0, -27375.0, -27435.0, -20400.0, -57180.0, -23330.0, -54900.0, -67095.0, -25475.0, -64220.0, -34365.0, -54330.0, -115515.0, -97800.0, -17635.0, -34440.0, -97035.0, -33110.0, -27380.0, -57160.0, -11610.0, -71705.0, -8630.0, -100870.0, -28270.0, -48040.0, -89985.0, -19610.0, -34440.0, -50765.0, -13610.0, -53195.0, -38095.0, -40060.0, -17450.0, -49825.0, -10320.0, -18550.0, -93140.0, -18640.0, -68185.0, -87110.0, -34385.0, -28415.0, -64385.0, -45020.0, -51770.0, -68100.0, -20400.0, -104185.0, -26530.0, -12240.0, -200130.0, -29460.0, -34990.0, -32315.0, -120010.0, -16405.0, -34385.0, -42350.0, -269600.0, -44080.0, -61130.0, -9790.0, -30310.0, -30425.0, -30445.0, -18335.0, -70745.0, -27425.0, -69990.0, -67055.0, -22295.0, -11530.0, -26345.0, -31455.0, -29450.0, -23505.0, -25660.0, -33265.0, -22525.0, -89160.0, -20435.0, -21685.0, -37180.0, -51340.0, -29550.0, -38400.0, -60395.0, -14590.0, -52335.0, -28365.0, -38255.0, -40340.0, -56200.0, -29540.0, -47385.0, -17605.0, -20640.0, -92360.0, -17575.0, -27560.0, -42440.0, -45745.0, -67040.0, -48340.0, -40360.0, -41355.0, -117355.0, -35180.0, -118880.0, -44305.0, -25670.0, -30450.0, -58060.0, -38440.0, -44320.0, -28575.0, -61105.0, -42630.0, -33620.0, -60040.0, -52380.0, -20475.0, -35735.0, -29480.0, -37525.0, -57875.0, -32790.0, -34545.0, -32705.0, -12810.0, -16830.0, -16785.0, -48480.0, -22410.0, -23745.0, -19815.0, -24830.0, -29685.0, -30070.0, -34780.0, -42655.0, -32680.0, -20695.0, -26430.0, -26815.0, -22645.0, -21760.0, -16825.0, -34120.0, -21510.0, -26775.0, -13320.0, -33675.0, -21760.0, -23615.0, -13770.0, -11830.0, -28695.0, -11855.0, -19440.0, -14460.0, -31405.0, -65420.0, -22660.0, -28725.0, -14650.0, -25780.0, -21845.0, -33645.0, -37665.0, -20515.0, -34735.0, -36670.0, -23720.0, -40555.0, -20805.0, -35605.0, -27615.0, -19520.0, -16830.0, -20470.0, -17685.0, -17240.0, -19565.0, -34780.0, -22650.0, -34780.0, -21665.0, -36630.0, -30755.0, -30755.0, -11785.0, -39640.0, -18820.0, -30620.0, -32790.0, -27775.0, -55520.0, -12775.0, -19710.0, -23835.0, -22625.0, -25665.0, -10820.0, -10790.0, -12775.0, -30560.0, -21805.0, -21545.0, -49460.0, -12460.0, -25610.0, -12730.0, -29765.0, -28440.0, -50985.0, -19615.0, -44450.0, -51325.0, -43060.0, -54420.0, -21090.0, -114710.0, -19175.0, -180220.0, -33335.0, -196245.0, -18290.0, -13110.0, -166625.0, -100100.0, -266945.0, -220165.0, -21550.0, -34200.0, -53680.0, -25755.0, -55095.0, -134390.0, -18255.0, -63725.0, -156770.0, -124525.0, -40420.0, -34270.0, -149465.0, -111360.0, -180860.0, -29400.0, -44130.0, -35650.0, -86785.0, -38295.0, -19705.0, -56585.0, -108980.0, -177230.0, -20480.0, -252765.0, -47225.0, -65970.0, -73770.0, -72345.0, -59305.0, -16680.0, -31255.0, -26645.0, -133325.0, -20235.0, -21315.0, -189700.0, -22780.0, -16365.0, -138685.0, -120625.0, -231735.0, -143600.0, -234590.0, -34650.0, -138685.0, -164700.0, -182680.0, -128665.0, -109570.0, -91480.0, -98515.0, -233605.0, -117610.0, -200855.0, -136705.0, -106555.0, -167645.0, -94495.0, -176905.0, -86455.0, -150640.0, -121525.0, -170595.0, -129630.0, -143690.0, -86455.0, -135700.0, -70375.0, -117610.0, -110575.0, -92485.0, -151780.0, -140725.0, -113590.0, -81430.0, -132685.0, -95500.0, -10990.0]

iterations = range(0, len(rews), 1)
plt.plot(iterations, rews)
# plt.plot(iterations, ep_rew2, label='Dueling + Prioritized Replay')
# plt.plot(iterations, ep_rew1, label='Double DQN')
# plt.plot(iterations, rews, label='DQN')
# plt.plot(iterations, three_rew)
# plt.legend()
# plt.xlabel("Number of timesteps")
plt.ylabel("Average Return")



# iterations = range(0, len(col3), 1)
# plt.plot(iterations, three_col)
# plt.plot(iterations, col2, label='Dueling + Prioritized Replay')
# plt.plot(iterations, col1, label='Double DQN')
# plt.plot(iterations, col3, label='DQN')
# plt.legend()
plt.xlabel("Number of Episodes")
# plt.ylabel("Number of Collisions")
plt.show()
