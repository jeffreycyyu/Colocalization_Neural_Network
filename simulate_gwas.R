library(simGWAS)


#set empty list to later bind all samples with
list_of_samples = list()

n_samples = 100

for (i in 1:n_samples) {
  
  set.seed(i)
  nsnps <- 100
  nhaps <- 1000
  lag <- 5 # genotypes are correlated between neighbouring variants
  maf <- runif(nsnps+lag,0.05,0.5) # common SNPs
  laghaps <- do.call("cbind", lapply(maf, function(f) rbinom(nhaps,1,f)))
  haps <- laghaps[,1:nsnps]
  for(j in 1:lag) 
    haps <- haps + laghaps[,(1:nsnps)+j]
  haps <- round(haps/matrix(apply(haps,2,max),nhaps,nsnps,byrow=TRUE))
  
  snps <- colnames(haps) <- paste0("s",1:nsnps)
  freq <- as.data.frame(haps+1)
  freq$Probability <- 1/nrow(freq)
  sum(freq$Probability)
  
  CV=sample(snps[which(colMeans(haps)>0.1)],2)
  g1 <- c(1.4,1.2)
  
  
  FP <- make_GenoProbList(snps=snps,W=CV,freq=freq)
  zexp <- expected_z_score(N0=10000, # number of controls
                           N1=10000, # number of cases
                           snps=snps, # column names in freq of SNPs for which Z scores should be generated
                           W=CV, # causal variants, subset of snps
                           gamma.W=log(g1), # odds ratios
                           freq=freq, # reference haplotypes
                           GenoProbList=FP) # FP above
  
  
  plot(1:nsnps,zexp); abline(v=which(snps %in% CV),col="red"); abline(h=0)
  
  
  # 3 replicates of the same pattern for this seed
  zsim_1 <- simulated_z_score(N0=10000, # number of controls
                            N1=10000, # number of cases
                            snps=snps, # column names in freq of SNPs for which Z scores should be generated
                            W=CV, # causal variants, subset of snps
                            gamma.W=log(g1), # log odds ratios
                            freq=freq, # reference haplotypes
                            nrep=3)
  
  zsim_2 <- simulated_z_score(N0=10000, # number of controls
                              N1=10000, # number of cases
                              snps=snps, # column names in freq of SNPs for which Z scores should be generated
                              W=CV, # causal variants, subset of snps
                              gamma.W=log(g1), # log odds ratios
                              freq=freq, # reference haplotypes
                              nrep=3)
  
  zsim_3 <- simulated_z_score(N0=10000, # number of controls
                              N1=10000, # number of cases
                              snps=snps, # column names in freq of SNPs for which Z scores should be generated
                              W=CV, # causal variants, subset of snps
                              gamma.W=log(g1), # log odds ratios
                              freq=freq, # reference haplotypes
                              nrep=3)
  
  
  
  # plot z-scores
  # par(mfcol=c(3,1))
  # for(i in 1:3) {
  #   plot(1:nsnps,zexp,xlab="SNP",ylab="Z score"); abline(v=which(snps %in% CV),col="red"); abline(h=0)
  #   title(main=paste("Replication",i))
  #   points(1:nsnps,zsim[i,],col="blue",pch=2)
  # } 
  # 
  
  
  
  
  
  
  #function for 10^-x
  ten_to_negative_power <- function(input_z_score) {
    output_p_value <- 10^(-input_z_score)
    return(output_p_value)
  }
  
  #turn simulated z-scores into the 10^-x scale for p-values
  simulated_p_values_informal_1 <- lapply(zsim_1, ten_to_negative_power)
  simulated_p_values_informal_2 <- lapply(zsim_2, ten_to_negative_power)
  simulated_p_values_informal_3 <- lapply(zsim_3, ten_to_negative_power)
  
  
  #convert all above 1 to 0.99
  output_p_value_formal_1 <- ifelse(simulated_p_values_informal_1 >= 1, 0.99, simulated_p_values_informal_1)
  output_p_value_formal_2 <- ifelse(simulated_p_values_informal_2 >= 1, 0.99, simulated_p_values_informal_2)
  output_p_value_formal_3 <- ifelse(simulated_p_values_informal_3 >= 1, 0.99, simulated_p_values_informal_3)
  # plot(1:300, output_p_value_formal_1)
  # plot(1:300, output_p_value_formal_2)
  # plot(1:300, output_p_value_formal_3)
  
  
  
  #add to list
  list_of_samples[[i]] <- cbind(unlist(output_p_value_formal_1), unlist(output_p_value_formal_2), unlist(output_p_value_formal_3))
  
}

list_of_samples_binded <- do.call(rbind, list_of_samples)



#save as data table
df <- data.frame(c(1:(300*n_samples)), list_of_samples_binded)

write.table(df,"simulated_gwas.txt",sep="\t",row.names=FALSE)







# 
# 
# 
# #sanity check
# #function for -log10(x)
# ten_to_negative_power_opposite <- function(input_z_score) {
#   output_p_value <- -log10(input_z_score)
#   return(output_p_value)
# }
# #turn back to "z"-scores
# simulated_p_values_informal_opposite <- lapply(list_of_samples_semi_formal, ten_to_negative_power_opposite)
# 
# plot(1:300, simulated_p_values_informal_opposite)
# 














# library(simGWAS)
# 
# set.seed(1)
# nsnps <- 1000
# nhaps <- 10000
# lag <- 5 # genotypes are correlated between neighbouring variants
# maf <- runif(nsnps+lag,0.05,0.5) # common SNPs
# laghaps <- do.call("cbind", lapply(maf, function(f) rbinom(nhaps,1,f)))
# haps <- laghaps[,1:nsnps]
# for(j in 1:lag) 
#   haps <- haps + laghaps[,(1:nsnps)+j]
# haps <- round(haps/matrix(apply(haps,2,max),nhaps,nsnps,byrow=TRUE))
# 
# snps <- colnames(haps) <- paste0("s",1:nsnps)
# freq <- as.data.frame(haps+1)
# freq$Probability <- 1/nrow(freq)
# sum(freq$Probability)
# 
# 
# #pick 75 CVs
# CV=sample(snps[which(colMeans(haps)>0.1)],75)
# g1 <- runif(75, 1.1, 5)
# 
# 
# FP <- make_GenoProbList(snps=snps,W=CV,freq=freq)
# zexp <- expected_z_score(N0=100, # number of controls
#                          N1=100, # number of cases
#                          snps=snps, # column names in freq of SNPs for which Z scores should be generated
#                          W=CV, # causal variants, subset of snps
#                          gamma.W=log(g1), # odds ratios
#                          freq=freq, # reference haplotypes
#                          GenoProbList=FP) # FP above
# 
# 
# plot(1:nsnps,zexp); abline(v=which(snps %in% CV),col="red"); abline(h=0)
# 


