# MMD（Maximum Mean Discrepancy）

基于两个分布的样本，通过寻找在样本空间上的连续函数f，求不同分布的样本在f上的函数值的均值，通过把两个均值作差可以得到两个分布对应于f的mean discrepancy。寻找一个f使得这个mean discrepancy有最大值，就得到了MMD。最后取MMD作为检验统计量（test statistic），从而判断两个分布是否相同。
如果这个值足够小，就认为两个分布相同，否则就认为它们不相同。同时这个值也用来判断两个分布之间的相似程度。